import mujoco
import numpy as np
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
##

from .metrics import BodyName, JointName, Phase, FailureMode
from .metrics import PhaseSnapshot, ContactTimeSeries, OrientationTimeSeries, ObjState

from utils.utils import unit, angular_deviation, safe_divide

# -------------------- Helpers --------------------

def _phase_tail_contacts(contacts: "ContactTimeSeries",
                         phase_name: str,
                         tail_frac: float = 0.2) -> int:
    """
    Median # of finger-target contacts over the last `tail_frac` of the samples
    tagged with `phase_name`. Robust to single-step contact chatter that can
    make a single-instant snapshot misleadingly read 0 on a grip that was
    actually stable for the rest of the phase.
    """
    n_total = len(contacts)
    if n_total == 0:
        return 0
    indices = [i for i in range(n_total) if contacts.phase[i] == phase_name]
    if not indices:
        return 0
    k = max(1, int(len(indices) * tail_frac))
    tail = [contacts.n_contacts[i] for i in indices[-k:]]
    return int(np.median(tail))

def _phase_boundaries(t: np.ndarray, phase_list) -> List[Tuple[float, str]]:
    """Return [(t_at_boundary, phase_name)] each time the phase label changes."""
    out, last = [], None
    for i, p in enumerate(phase_list):
        if p != last:
            out.append((float(t[i]), str(p)))
            last = p
    return out

def _decorate_axes(axes, boundaries):
    """Vertical phase markers + grid + rotated phase labels in the top axis."""
    for ax in axes:
        for tb, _ in boundaries:
            ax.axvline(tb, color="gray", alpha=0.4, linestyle="-.")
        ax.grid(alpha=0.3)
    if not boundaries:
        return
    top = axes[0]
    ymax = top.get_ylim()[1]
    if ymax <= 0:
        ymax = 1.0
    for tb, ph in boundaries:
        top.text(tb, ymax * 0.95, ph, rotation=90, fontsize=7,
                 va="top", ha="right")


# -------------------- Grasp Record Class --------------------

@dataclass
class GraspRecord:
    """Full per-grasp record."""
    grasp_index: int
    score:       float
    object_id:   int
    target_body: str
    target_bid:  int
    t_world:     np.ndarray
    width:       float
    approach_baseline: Optional[np.ndarray] = None   # commanded approach in world frame
    binormal_baseline: Optional[np.ndarray] = None   # commanded binormal in world frame
    checkpoints:     Dict[str, PhaseSnapshot] = field(default_factory=dict)
    contacts:        ContactTimeSeries        = field(default_factory=ContactTimeSeries)
    orientation:     OrientationTimeSeries    = field(default_factory=OrientationTimeSeries)
    lifted_objects:  List[Tuple[str, float]]  = field(default_factory=list)
    _success:        Optional[bool]           = None

    def finalize(self, success: bool, lifted_objects):
        self._success = bool(success)
        self.lifted_objects = list(lifted_objects)

    @property
    def success(self) -> bool:
        return bool(self._success)

    @property
    def state(self) -> ObjState:
        return ObjState(
            seated=_phase_tail_contacts(self.contacts, Phase.CLOSE.value)   > 0, # After it closed the fingers, the last number contacts 
            held  =_phase_tail_contacts(self.contacts, Phase.RETREAT.value) > 0, # After it retreated, 
            lifted=self.success,
        )

    @property
    def failure_mode(self) -> FailureMode:
        return self.state.failure_mode


# -------------------- Evaluator --------------------

class GraspEvaluator:
    """
    Optional per-grasp instrumentation. Three features:
      (1) Per-step finger↔target-object contact state (count, fn/ft per finger).
      (2) Per-step hand-axis orientation (approach + binormal in world).
      (3) Named phase checkpoints + GraspRecord aggregation.

    Lifecycle (driven by GraspHandMocap):
        begin_grasp(...) -> mark_phase(Phase.X)* -> log_step()* -> end_grasp(...)
    When the evaluator is not attached, those calls are skipped at the call site.
    """

    def __init__(self, model, data, mu_estimate: float = 1.0):
        self.model = model
        self.data  = data
        self.mu    = float(mu_estimate)

        self.finger_bids: Dict[BodyName, int] = {}
        for n in BodyName.fingers():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n.value)
            if bid == -1:
                raise RuntimeError(f"Finger body '{n}' not found in model.")
            self.finger_bids[n] = bid

        self._hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BodyName.HAND.value)
        self._fj1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JointName.FINGER_JOINT1.value)
        self._fj2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JointName.FINGER_JOINT2.value)

        self.records: List[GraspRecord] = []
        self._current: Optional[GraspRecord] = None
        self._phase: str = Phase.INIT.value
        self._force6 = np.zeros(6)

    def reset(self):
        self.records = []
        self._current = None
        self._phase = Phase.INIT.value

    # ---------------- lifecycle ----------------

    def begin_grasp(self, grasp_index, score, object_id, t_world, width,
                    approach_baseline=None, binormal_baseline=None):
        target_name = f"obj_{int(object_id):03d}"
        target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        self._current = GraspRecord(
            grasp_index=int(grasp_index),
            score=float(score),
            object_id=int(object_id),
            target_body=target_name,
            target_bid=int(target_bid),
            t_world=np.asarray(t_world, dtype=np.float64),
            width=float(width),
            approach_baseline=unit(approach_baseline),
            binormal_baseline=unit(binormal_baseline),
        )
        self._phase = Phase.INIT.value

    def mark_phase(self, phase_name):
        """Boundary marker: snapshots state, then tags subsequent step logs."""
        self._phase = str(phase_name)
        if self._current is None:
            return
        self._current.checkpoints[self._phase] = self._snapshot()

    def log_step(self):
        """Called every sim step while a grasp is active. Logs contacts + hand orientation."""
        if self._current is None or self._current.target_bid == -1:
            return

        left_bid   = self.finger_bids[BodyName.LEFT_FINGER]
        right_bid  = self.finger_bids[BodyName.RIGHT_FINGER]
        target_bid = self._current.target_bid
        geom_body  = self.model.geom_bodyid

        left_fn = left_ft = right_fn = right_ft = 0.0
        n_pairs = 0

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = int(geom_body[con.geom1])
            b2 = int(geom_body[con.geom2])
            if target_bid not in (b1, b2):
                continue
            finger_bid = b1 if b1 in (left_bid, right_bid) else (
                         b2 if b2 in (left_bid, right_bid) else -1)
            if finger_bid == -1:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, self._force6)
            fn = abs(float(self._force6[0]))
            ft = float(np.linalg.norm(self._force6[1:3]))
            if finger_bid == left_bid:
                left_fn += fn; left_ft += ft
            else:
                right_fn += fn; right_ft += ft
            n_pairs += 1

        t_now = float(self.data.time)
        self._current.contacts.append_sample(
            t=t_now, phase=self._phase, n_contacts=n_pairs,
            left_fn=left_fn, left_ft=left_ft,
            right_fn=right_fn, right_ft=right_ft,
            finger_gap=self._finger_gap(),
        )
        approach_w, binormal_w = self._hand_axes_world()
        self._current.orientation.append_sample(
            t=t_now, phase=self._phase,
            approach=approach_w, binormal=binormal_w,
        )

    def end_grasp(self, success, lifted):
        if self._current is None:
            return
        self._current.finalize(success=success, lifted_objects=lifted)
        self.records.append(self._current)
        self._current = None
        self._phase = Phase.INIT.value

    # ---------------- internals ----------------

    def _snapshot(self) -> PhaseSnapshot:
        target_bid = self._current.target_bid
        obj_pos = obj_quat = None
        if target_bid != -1:
            obj_pos  = self.data.xpos[target_bid].copy()
            obj_quat = self.data.xquat[target_bid].copy()
        hand_pos = (self.data.xpos[self._hand_bid].copy()
                    if self._hand_bid != -1 else np.zeros(3))
        return PhaseSnapshot(
            t=float(self.data.time),
            phase=self._phase,
            finger_gap=self._finger_gap(),
            n_finger_obj_contacts=self._count_finger_target_contacts(target_bid),
            hand_pos=hand_pos,
            obj_pos=obj_pos,
            obj_quat=obj_quat,
        )

    def _count_finger_target_contacts(self, target_bid: int) -> int:
        """
        Count current contacts between any finger and the target body.
        """
        if target_bid == -1:
            return 0
        left_bid  = self.finger_bids[BodyName.LEFT_FINGER]
        right_bid = self.finger_bids[BodyName.RIGHT_FINGER]
        geom_body = self.model.geom_bodyid
        count = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = int(geom_body[con.geom1])
            b2 = int(geom_body[con.geom2])
            if target_bid in (b1, b2) and (left_bid in (b1, b2) or right_bid in (b1, b2)):
                count += 1
        return count

    def _finger_gap(self) -> float:
        if self._fj1 == -1 or self._fj2 == -1:
            return float("nan")
        q1 = float(self.data.qpos[self.model.jnt_qposadr[self._fj1]])
        q2 = float(self.data.qpos[self.model.jnt_qposadr[self._fj2]])
        return q1 + q2

    def _hand_axes_world(self):
        """
        Hand body rotation: columns of xmat are local axes expressed in world.
        Per to_mujoco_quat(): local Z = approach, local Y = binormal.
        """
        if self._hand_bid == -1:
            zero = np.zeros(3)
            return zero, zero
        R = self.data.xmat[self._hand_bid].reshape(3, 3)
        return R[:, 2].copy(), R[:, 1].copy()

    # ---------------- summary ----------------

    def summary_table(self):
        """Per-object funnel: tested → seated → held → lifted, with dominant failure mode."""
        agg = defaultdict(lambda: {"n": 0, "seated": 0, "held": 0, "lifted": 0,
                                   "modes": defaultdict(int)})
        for r in self.records:
            st = r.state
            row = agg[r.object_id]
            row["n"]      += 1
            row["seated"] += int(st.seated)
            row["held"]   += int(st.held)
            row["lifted"] += int(st.lifted)
            row["modes"][str(st.failure_mode)] += 1

        rows = []
        for obj_id in sorted(agg.keys()):
            v = agg[obj_id]
            dom = max(v["modes"].items(), key=lambda kv: kv[1])[0] if v["modes"] else "-"
            rows.append({
                "object_id":     obj_id,
                "n_grasps":      v["n"],
                "seated":        v["seated"],
                "held":          v["held"],
                "lifted":        v["lifted"],
                "dominant_mode": dom,
            })
        return rows

    def print_summary(self, log=print):
        rows = self.summary_table()
        if not rows:
            log("[grasp-eval] no records.")
            return
        header = f"{'obj_id':>6} {'n':>4} {'seated':>7} {'held':>5} {'lifted':>7} {'dominant_failure':>26}"
        log(header)
        log("-" * len(header))
        for r in rows:
            log(f"{r['object_id']:>6} {r['n_grasps']:>4} {r['seated']:>7} "
                f"{r['held']:>5} {r['lifted']:>7} {r['dominant_mode']:>26}")

    # ---------------- I/O ----------------

    # Phases that get their own column block in the per-grasp CSV.
    PHASES_FOR_COLS = (Phase.APPROACH, Phase.CLOSE, Phase.RETREAT, Phase.LIFT)

    def save_performance_csv(self, path: str):
        """
        ONE ROW PER GRASP. Columns:
          - identity:   grasp_index, object_id, score, target_body, width
          - per phase:  {phase}_max_dev_approach_deg, {phase}_max_dev_binormal_deg,
                        {phase}_peak_fn, {phase}_end_contacts
          - outcome:    seated, held, lifted, success, failure_mode
        """
        if not self.records:
            return
        rows = [self._grasp_row(r) for r in self.records]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _grasp_row(self, rec: GraspRecord) -> Dict:
        c, o = rec.contacts, rec.orientation
        n = min(len(c), len(o))
        A = np.asarray(o.approach[:n]) if n > 0 else np.zeros((0, 3))
        B = np.asarray(o.binormal[:n]) if n > 0 else np.zeros((0, 3))
        dev_a    = angular_deviation(A, rec.approach_baseline)
        dev_b    = angular_deviation(B, rec.binormal_baseline)
        fn_total = (np.asarray(c.left_fn[:n]) + np.asarray(c.right_fn[:n])
                    if n > 0 else np.zeros(0))
        phases   = np.asarray(c.phase[:n])

        row: Dict = {
            "grasp_index": rec.grasp_index,
            "object_id":   rec.object_id,
            "score":       rec.score,
            "target_body": rec.target_body,
            "width":       rec.width,
        }
        for ph in self.PHASES_FOR_COLS:
            key  = ph.value
            mask = phases == key
            if mask.any():
                row[f"{key}_max_dev_approach_deg"] = float(dev_a[mask].max())
                row[f"{key}_max_dev_binormal_deg"] = float(dev_b[mask].max())
                row[f"{key}_peak_fn"]              = float(fn_total[mask].max())
            else:
                row[f"{key}_max_dev_approach_deg"] = float("nan")
                row[f"{key}_max_dev_binormal_deg"] = float("nan")
                row[f"{key}_peak_fn"]              = float("nan")
            row[f"{key}_end_contacts"] = _phase_tail_contacts(c, key)

        st = rec.state
        row.update({
            "seated":       int(st.seated),
            "held":         int(st.held),
            "lifted":       int(st.lifted),
            "success":      int(rec.success),
            "failure_mode": str(rec.failure_mode),
        })
        return row

    # ---------------- plotting ----------------

    def plot_grasp(self, idx: int, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        rec = self.records[idx]
        c = rec.contacts
        if len(c) == 0:
            return None

        t = np.asarray(c.t)
        boundaries = _phase_boundaries(t, c.phase)

        l_fn = np.asarray(c.left_fn);  r_fn = np.asarray(c.right_fn)
        l_ft = np.asarray(c.left_ft);  r_ft = np.asarray(c.right_ft)
        fn_sum = l_fn + r_fn
        ft_sum = l_ft + r_ft
        ratio = safe_divide(ft_sum, fn_sum)

        fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        axes[0].plot(t, c.n_contacts, "k-")
        axes[0].set_ylabel("# finger-obj contacts")
        axes[1].plot(t, l_fn, label="L  fn"); axes[1].plot(t, r_fn, label="R  fn")
        axes[1].plot(t, l_ft, "--", label="L  ft"); axes[1].plot(t, r_ft, "--", label="R  ft")
        axes[1].set_ylabel("force [N]"); axes[1].legend(fontsize=8)
        axes[2].plot(t, ratio, "m-", label="|ft| / |fn|")
        axes[2].axhline(self.mu, color="k", linestyle=":", label=f"mu={self.mu:.2f}")
        axes[2].set_ylabel("friction-cone ratio"); axes[2].legend(fontsize=8)
        axes[3].plot(t, c.finger_gap, "b-")
        axes[3].axhline(rec.width, color="r", linestyle=":",
                        label=f"grasp width={rec.width:.3f}")
        axes[3].set_ylabel("finger gap [m]"); axes[3].set_xlabel("time [s]")
        axes[3].legend(fontsize=8)

        _decorate_axes(axes, boundaries)
        fig.suptitle(f"grasp #{rec.grasp_index}  obj={rec.object_id:03d}  "
                     f"score={rec.score:.3f}  →  {rec.failure_mode}")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        return fig

    def plot_grasp_orientation(self, idx: int, save_path: Optional[str] = None):
        """
        Per-grasp hand-axis drift. Three panels:
          (1) approach world components (x,y,z) actual vs baseline horizontal lines
          (2) binormal world components (x,y,z) actual vs baseline horizontal lines
          (3) angular deviation [deg] of both axes from their baselines
        Phase boundaries are overlaid as vertical lines.
        """
        import matplotlib.pyplot as plt
        rec = self.records[idx]
        o = rec.orientation
        if len(o) == 0:
            return None

        t, A, B = o.as_arrays()
        boundaries = _phase_boundaries(t, o.phase)

        a_base, b_base = rec.approach_baseline, rec.binormal_baseline
        dev_a = angular_deviation(A, a_base)
        dev_b = angular_deviation(B, b_base)

        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        colors = ("tab:red", "tab:green", "tab:blue")
        for k, lbl in enumerate("xyz"):
            axes[0].plot(t, A[:, k], color=colors[k], label=f"approach.{lbl}")
            if a_base is not None:
                axes[0].axhline(a_base[k], color=colors[k], linestyle=":",
                                alpha=0.7, label=f"baseline.{lbl}")
        axes[0].set_ylabel("approach axis (unit)")
        axes[0].legend(fontsize=7, ncol=2, loc="upper right")

        for k, lbl in enumerate("xyz"):
            axes[1].plot(t, B[:, k], color=colors[k], label=f"binormal.{lbl}")
            if b_base is not None:
                axes[1].axhline(b_base[k], color=colors[k], linestyle=":",
                                alpha=0.7, label=f"baseline.{lbl}")
        axes[1].set_ylabel("binormal axis (unit)")
        axes[1].legend(fontsize=7, ncol=2, loc="upper right")

        axes[2].plot(t, dev_a, color="tab:purple", label="approach dev")
        axes[2].plot(t, dev_b, color="tab:orange", label="binormal dev")
        axes[2].set_ylabel("angular dev from baseline [deg]")
        axes[2].set_xlabel("time [s]")
        axes[2].legend(fontsize=8)

        _decorate_axes(axes, boundaries)
        fig.suptitle(f"grasp #{rec.grasp_index}  obj={rec.object_id:03d}  "
                     f"score={rec.score:.3f}  →  orientation drift")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        return fig

    def plot_all(self, out_dir: str):
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        for i in range(len(self.records)):
            fig = self.plot_grasp(i, save_path=os.path.join(out_dir, f"grasp_{i:03d}_contacts.png"))
            if fig is not None:
                plt.close(fig)
            fig = self.plot_grasp_orientation(i, save_path=os.path.join(out_dir, f"grasp_{i:03d}_orient.png"))
            if fig is not None:
                plt.close(fig)

    def plot_object_summary(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        rows = self.summary_table()
        if not rows:
            return None
        ids   = [str(r["object_id"]) for r in rows]
        n     = np.array([r["n_grasps"] for r in rows])
        seat  = np.array([r["seated"]   for r in rows])
        held  = np.array([r["held"]     for r in rows])
        lift  = np.array([r["lifted"]   for r in rows])

        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(ids)), 4.5))
        x = np.arange(len(ids))
        w = 0.22
        ax.bar(x - 1.5 * w, n,    w, label="tested")
        ax.bar(x - 0.5 * w, seat, w, label="seated (post-close)")
        ax.bar(x + 0.5 * w, held, w, label="held (post-retreat)")
        ax.bar(x + 1.5 * w, lift, w, label="lifted (success)")
        ax.set_xticks(x); ax.set_xticklabels(ids)
        ax.set_xlabel("object_id"); ax.set_ylabel("# grasps")
        ax.set_title("per-object grasp funnel")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        return fig

    # ---------------- one-shot dump ----------------

    def dump(self, out_dir: str):
        """Write summary plot, per-grasp performance CSV, and per-grasp plots."""
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        self.save_performance_csv(os.path.join(out_dir, "performance.csv"))
        fig = self.plot_object_summary(os.path.join(out_dir, "summary.png"))
        if fig is not None:
            plt.close(fig)
        self.plot_all(os.path.join(out_dir, "per_grasp"))


