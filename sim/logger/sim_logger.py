import numpy as np
import mujoco


class SimLogger:
    """
    Debug logger for MuJoCo bodies. Records per-step kinematic state for a
    set of bodies (objects + hand) into in-memory buffers. No terminal output.

    Velocities for dynamic bodies come from `mj_objectVelocity` (world frame).
    The hand base in this project is kinematic (driven via model.body_pos /
    model.body_quat), so MuJoCo reports zero cvel for it — a finite-difference
    velocity is computed alongside so the hand trace is still meaningful.
    """

    FIELDS = ("t", "pos", "quat", "lin_vel", "ang_vel", "lin_vel_fd")

    def __init__(self, model, data, body_names=None, log_every=1):
        self.model = model
        self.data = data
        self.log_every = max(1, int(log_every))
        self._counter = 0

        if body_names is None:
            body_names = self._autodetect_bodies(model)

        self.body_names = list(body_names)
        self.body_ids = []
        missing = []
        for n in self.body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid == -1:
                missing.append(n)
            else:
                self.body_ids.append(bid)
        if missing:
            self.body_names = [n for n in self.body_names if n not in missing]

        self.records = {n: {k: [] for k in self.FIELDS} for n in self.body_names}
        self._prev = {n: None for n in self.body_names}  # (t, pos) for FD

    @staticmethod
    def _autodetect_bodies(model):
        names = []
        for i in range(100):
            n = f"obj_{i:03d}"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) != -1:
                names.append(n)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand") != -1:
            names.append("hand")
        return names

    def log(self):
        self._counter += 1
        if self._counter % self.log_every != 0:
            return

        t = float(self.data.time)
        vel6 = np.zeros(6)
        for n, bid in zip(self.body_names, self.body_ids):
            pos = self.data.xpos[bid].copy()
            quat = self.data.xquat[bid].copy()

            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0
            )
            ang_vel = vel6[0:3].copy()
            lin_vel = vel6[3:6].copy()

            prev = self._prev[n]
            if prev is None or t <= prev[0]:
                lin_vel_fd = np.zeros(3)
            else:
                lin_vel_fd = (pos - prev[1]) / (t - prev[0])
            self._prev[n] = (t, pos)

            rec = self.records[n]
            rec["t"].append(t)
            rec["pos"].append(pos)
            rec["quat"].append(quat)
            rec["lin_vel"].append(lin_vel)
            rec["ang_vel"].append(ang_vel)
            rec["lin_vel_fd"].append(lin_vel_fd)

    def reset(self):
        for rec in self.records.values():
            for k in rec:
                rec[k].clear()
        self._prev = {n: None for n in self.body_names}
        self._counter = 0

    def as_arrays(self, body_name):
        rec = self.records[body_name]
        return {k: np.array(v) for k, v in rec.items()}

    def save(self, path):
        arrs = {}
        for n in self.body_names:
            for k, v in self.records[n].items():
                arrs[f"{n}__{k}"] = np.array(v)
        np.savez(path, **arrs)

    def plot(self, body_name, save_path=None, use_fd_for_hand=True):
        import matplotlib.pyplot as plt

        d = self.as_arrays(body_name)
        if d["t"].size == 0:
            raise RuntimeError(f"no samples logged for {body_name}")

        t = d["t"]
        pos = d["pos"]
        lin = d["lin_vel_fd"] if (body_name == "hand" and use_fd_for_hand) else d["lin_vel"]
        ang = d["ang_vel"]

        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for i, lbl in enumerate("xyz"):
            axes[0].plot(t, pos[:, i], label=lbl)
            axes[1].plot(t, lin[:, i], label=lbl)
            axes[2].plot(t, ang[:, i], label=lbl)
        axes[0].set_ylabel("position [m]")
        vel_label = "lin vel (FD) [m/s]" if (body_name == "hand" and use_fd_for_hand) else "lin vel [m/s]"
        axes[1].set_ylabel(vel_label)
        axes[2].set_ylabel("ang vel [rad/s]")
        axes[2].set_xlabel("time [s]")
        for ax in axes:
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(body_name)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
        return fig

    def plot_all(self, out_dir):
        import os
        os.makedirs(out_dir, exist_ok=True)
        for n in self.body_names:
            if len(self.records[n]["t"]) == 0:
                continue
            self.plot(n, save_path=os.path.join(out_dir, f"{n}.png"))
