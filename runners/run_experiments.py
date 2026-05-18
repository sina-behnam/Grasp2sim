#!/usr/bin/env python3
"""
Grasp experiment runner: per-object isolation vs. combined-scene success rates,
with per-grasp evaluation instrumentation.

For each object found in the scene annotation:
  1. Individual  — scene contains only that object; top-N grasps are tested.
  2. Combined    — scene contains all objects; top-N grasps per object are tested
                   (success = the *target* object was lifted >= LIFT_HEIGHT).

Every grasp is classified into one of:
  ok                     — target object lifted
  no_contact_at_close    — fingers never seated on the target
  lost_during_retreat    — seated but slipped before the lift began
  slipped_during_lift    — held through retreat but lost during the lift
  wrong_object_lifted    — held the target through retreat, but a different
                           object was lifted (combined mode only)

Per-grasp contact + orientation plots are saved ONLY for the
"seated + held + not lifted" cases (the interesting failures).

Outputs written to <output_dir>/:
  results.csv                one row per grasp trial (with state columns)
  summary.json               per-object / per-experiment / overall stats
  individual/obj_XXX/        contacts+orient plots for interesting failures
  combined/obj_XXX/          same, for combined-scene runs
  obj_XXX_individual.mp4     per-object video (individual mode, if --video)
  combined.mp4               all grasps video (combined mode,  if --video)
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.append("/home/sbehnam/Project/grasp2sim")

import csv
import json
import tempfile
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from loguru import logger
from graspnetAPI import GraspGroup

from manage import config
from sim.core.simulator import GraspHandMocap, Executors
from sim.geval.metrics import FailureMode
from scenes.grasp2scene_mocap import Scene
from utils.poses import gg_filter_by_object_id, gg_filter_by_width, gg_filter_by_orthogonal_approach

SCENE_DIR   = config.paths.SCENE_DIR
MODEL_DIR   = config.paths.MODEL_DIR
HAND_ASSETS = config.paths.HAND_ASSETS
CAMERA      = config.paths.CAMERA
GRASPS_NPY  = config.paths.GRASPS_NPY
CAMERA_EXTR = config.paths.CAMERA_EXTR
CAMERA_POSE = config.paths.CAMERA_POSE
OUTPUT_DIR  = config.paths.OUTPUT_DIR
FRICTION_MU = config.sim.FRICTION_MU

MODE_OK                  = FailureMode.OK.value
MODE_NO_CONTACT_AT_CLOSE = FailureMode.NO_CONTACT_AT_CLOSE.value
MODE_LOST_DURING_RETREAT = FailureMode.LOST_DURING_RETREAT.value
MODE_SLIPPED_DURING_LIFT = FailureMode.SLIPPED_DURING_LIFT.value
MODE_WRONG_OBJECT_LIFTED = "wrong_object_lifted"

ALL_MODES = (
    MODE_OK, MODE_NO_CONTACT_AT_CLOSE, MODE_LOST_DURING_RETREAT,
    MODE_SLIPPED_DURING_LIFT, MODE_WRONG_OBJECT_LIFTED,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _prepare_grasps(gg: GraspGroup, obj_id: int, top_n: int) -> GraspGroup:
    """Filter by object ID and width, sort by score, return top-N."""
    g = gg_filter_by_object_id(gg, object_id=obj_id)
    g = gg_filter_by_width(g, width_threshold=0.08)
    g = gg_filter_by_orthogonal_approach(g,
                                         orthogonal_threshold=np.cos(np.radians(25)),
                                         table_to_cam=np.load(CAMERA_EXTR))
    g.sort_by_score()
    return g[:top_n]


def _make_sim(xml_path: str, camera_extr: str, camera_pose: str,
              video: bool, grasp_eval: bool,
              overlay_corner: str = 'top_left') -> GraspHandMocap:
    return GraspHandMocap(
        scene_xml=xml_path,
        camera_extr=camera_extr,
        camera_pose=camera_pose,
        render='human' if video else 'off',
        grasp_eval=grasp_eval,
        grasp_eval_mu=FRICTION_MU,
        overlay_corner=overlay_corner,
    )


def _classify(success: bool, seated: bool, held: bool, lifted_any: bool) -> str:
    """Outcome classification using the target-specific success flag."""
    if success:
        return MODE_OK
    if not seated:
        return MODE_NO_CONTACT_AT_CLOSE
    if not held:
        return MODE_LOST_DURING_RETREAT
    if lifted_any:
        return MODE_WRONG_OBJECT_LIFTED
    return MODE_SLIPPED_DURING_LIFT


def _plot_grasp(sim: GraspHandMocap, rank: int, plot_dir: Path):
    """Save contacts + orientation plots for the latest grasp run on `sim`."""
    if sim.grasp_evaluator is None or not sim.grasp_evaluator.records:
        return

    import matplotlib.pyplot as plt
    plot_dir.mkdir(parents=True, exist_ok=True)
    idx = len(sim.grasp_evaluator.records) - 1
    fig = sim.grasp_evaluator.plot_grasp(
        idx, save_path=str(plot_dir / f"grasp_{rank:03d}_contacts.png"))
    if fig is not None:
        plt.close(fig)
    fig = sim.grasp_evaluator.plot_grasp_orientation(
        idx, save_path=str(plot_dir / f"grasp_{rank:03d}_orient.png"))
    if fig is not None:
        plt.close(fig)


def _run_grasps(sim: GraspHandMocap, sampled: GraspGroup, executor,
                target_obj_name: str | None,
                experiment: str, obj_id: int,
                output_dir: Path) -> list:
    """
    Run every grasp in `sampled` through `sim.run_grasp`, capture state from the
    debugger, classify each outcome, and save plots for the interesting failures.
    """
    plot_dir = output_dir / experiment / f"obj_{obj_id:03d}"
    records = []
    for rank in range(len(sampled)):
        g = sampled[rank]
        t_w, lifted = sim.run_grasp(g, executor, grasp_index=rank)
        lifted_any = len(lifted) > 0
        success = (lifted_any if target_obj_name is None
                   else any(name == target_obj_name for name, _ in lifted))

        dbg = sim.grasp_evaluator
        if dbg is not None and dbg.records:
            st = dbg.records[-1].state
            seated, held = bool(st.seated), bool(st.held)
        else:
            seated = held = False

        mode = _classify(success, seated, held, lifted_any)

        records.append({
            'experiment':   experiment,
            'obj_id':       obj_id,
            'rank':         rank + 1,
            'score':        float(g.score),
            'width':        float(g.width),
            'success':      int(success),
            'seated':       int(seated),
            'held':         int(held),
            'lifted_any':   int(lifted_any),
            'failure_mode': mode,
            'lifted':       str([(n, round(h, 4)) for n, h in lifted]),
        })

        tag = MODE_OK if success else mode
        logger.info(f"    [{rank+1}/{len(sampled)}] score={g.score:.3f}  "
                    f"seated={int(seated)} held={int(held)} success={int(success)}  -> {tag}")

        _plot_grasp(sim, rank + 1, plot_dir)

    return records


# ── individual experiments ─────────────────────────────────────────────────────

def run_individual(scene: Scene, gg: GraspGroup, obj_ids: list,
                   top_n: int, output_dir: Path,
                   executor, video: bool, overlay_corner: str = 'top_left') -> list:
    """One scene per object; returns list of result dicts."""
    all_records = []
    for obj_id in obj_ids:
        logger.info(f"=== INDIVIDUAL  obj_{obj_id:03d} ===")

        sampled = _prepare_grasps(gg, obj_id, top_n)
        if len(sampled) == 0:
            logger.warning(f"No grasps for obj_{obj_id:03d} - skipping.")
            continue

        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                         mode='w', dir=output_dir) as fh:
            xml_path = fh.name
        try:
            scene.save_xml(xml_path, obj_indexes=[obj_id], coacd=True, strength='original')
            sim = _make_sim(xml_path, str(CAMERA_EXTR), str(CAMERA_POSE),
                            video=video, grasp_eval=True,
                            overlay_corner=overlay_corner)
            sim.set_overlay_prefix("individual")

            logger.info(f"  Testing {len(sampled)} grasps")
            records = _run_grasps(sim, sampled, executor,
                                  target_obj_name=None,
                                  experiment='individual',
                                  obj_id=obj_id,
                                  output_dir=output_dir)

            if video:
                vpath = output_dir / f"obj_{obj_id:03d}_individual.mp4"
                sim.save_video(str(vpath), fps=5)
                logger.info(f"  Video -> {vpath}")
        finally:
            os.unlink(xml_path)

        n_ok = sum(1 for r in records if r['success'])
        logger.info(f"  obj_{obj_id:03d} individual: {n_ok}/{len(records)}")
        all_records.extend(records)

    return all_records


# ── combined experiment ────────────────────────────────────────────────────────

def run_combined(scene: Scene, gg: GraspGroup, obj_ids: list,
                 top_n: int, output_dir: Path,
                 executor, video: bool, overlay_corner: str = 'top_left') -> list:
    """All objects in one scene; test top-N grasps per object."""
    logger.info("=== COMBINED  (all objects together) ===")

    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                     mode='w', dir=output_dir) as fh:
        xml_path = fh.name
    try:
        scene.save_xml(xml_path, obj_indexes=None, coacd=True, strength='original')
        sim = _make_sim(xml_path, str(CAMERA_EXTR), str(CAMERA_POSE),
                        video=video, grasp_eval=True,
                        overlay_corner=overlay_corner)
        sim.set_overlay_prefix("combined")

        all_records = []
        for obj_id in obj_ids:
            sampled = _prepare_grasps(gg, obj_id, top_n)
            if len(sampled) == 0:
                logger.warning(f"No grasps for obj_{obj_id:03d} in combined - skipping.")
                continue

            logger.info(f"  obj_{obj_id:03d}: {len(sampled)} grasps")
            target_name = f"obj_{obj_id:03d}"
            records = _run_grasps(sim, sampled, executor,
                                  target_obj_name=target_name,
                                  experiment='combined',
                                  obj_id=obj_id,
                                  output_dir=output_dir)

            n_ok = sum(1 for r in records if r['success'])
            logger.info(f"  obj_{obj_id:03d} combined:    {n_ok}/{len(records)}")
            all_records.extend(records)

        if video:
            vpath = output_dir / "combined.mp4"
            sim.save_video(str(vpath), fps=5)
            logger.info(f"  Video -> {vpath}")
    finally:
        os.unlink(xml_path)

    return all_records


# ── summary ────────────────────────────────────────────────────────────────────

def _empty_bucket() -> dict:
    return {'total': 0, 'seated': 0, 'held': 0, 'lifted': 0,
            'modes': {m: 0 for m in ALL_MODES}}


def _accumulate(bucket: dict, r: dict):
    bucket['total']  += 1
    bucket['seated'] += int(r['seated'])
    bucket['held']   += int(r['held'])
    bucket['lifted'] += int(r['success'])
    bucket['modes'][r['failure_mode']] += 1


def _finish(s: dict) -> dict:
    t = s['total']
    return {
        'total':        t,
        'seated':       s['seated'],
        'held':         s['held'],
        'lifted':       s['lifted'],
        'seat_rate':    round(s['seated'] / t, 3) if t else 0.0,
        'hold_rate':    round(s['held']   / t, 3) if t else 0.0,
        'success_rate': round(s['lifted'] / t, 3) if t else 0.0,
        'modes':        {m: n for m, n in s['modes'].items() if n},
    }


def _summarize(records: list) -> dict:
    per_object     = defaultdict(_empty_bucket)
    per_experiment = defaultdict(_empty_bucket)
    overall        = _empty_bucket()

    for r in records:
        key = f"{r['experiment']}/obj_{r['obj_id']:03d}"
        _accumulate(per_object[key],         r)
        _accumulate(per_experiment[r['experiment']], r)
        _accumulate(overall,                 r)

    return {
        'per_object':     {k: _finish(v) for k, v in sorted(per_object.items())},
        'per_experiment': {k: _finish(v) for k, v in sorted(per_experiment.items())},
        'overall':        _finish(overall),
    }


def _modes_string(modes: dict) -> str:
    """Render the mode counts excluding 'ok' (covered by success_rate)."""
    return ", ".join(f"{m}={n}" for m, n in modes.items() if m != MODE_OK)


def _print_summary(summary: dict):
    logger.info("")
    logger.info("─── per-experiment / per-object ──────────────────────────────────────────")
    header = f"{'key':<28} {'n':>4} {'seat':>4} {'held':>4} {'lift':>4} {'rate':>6}  | failures"
    logger.info(header)
    logger.info("-" * (len(header) + 20))
    for key, s in summary['per_object'].items():
        logger.info(
            f"{key:<28} {s['total']:>4} {s['seated']:>4} {s['held']:>4} "
            f"{s['lifted']:>4} {100*s['success_rate']:5.1f}%  | {_modes_string(s['modes'])}"
        )

    logger.info("")
    logger.info("─── per-experiment totals ────────────────────────────────────────────────")
    for exp, s in summary['per_experiment'].items():
        logger.info(
            f"  {exp:<10} n={s['total']:>4}  "
            f"success={100*s['success_rate']:5.1f}%  "
            f"seat={100*s['seat_rate']:5.1f}%  hold={100*s['hold_rate']:5.1f}%  | "
            f"{_modes_string(s['modes'])}"
        )

    o = summary['overall']
    logger.info("")
    logger.info(
        f"─── OVERALL ─── n={o['total']}  "
        f"success={100*o['success_rate']:.1f}%  "
        f"seat={100*o['seat_rate']:.1f}%  hold={100*o['hold_rate']:.1f}%"
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global GRASPS_NPY, CAMERA_EXTR, CAMERA_POSE

    parser = argparse.ArgumentParser(
        description="Run grasp experiments: per-object isolation vs. combined scene.")
    parser.add_argument("--scene-dir",     default=SCENE_DIR)
    parser.add_argument("--model-dir",     default=MODEL_DIR)
    parser.add_argument("--hand-assets",   default=HAND_ASSETS)
    parser.add_argument("--camera",        default=CAMERA)
    parser.add_argument("--grasps-npy",    default=GRASPS_NPY)
    parser.add_argument("--camera-extr",   default=CAMERA_EXTR)
    parser.add_argument("--camera-pose",   default=CAMERA_POSE)
    parser.add_argument("--output-dir",    default=OUTPUT_DIR)
    parser.add_argument("--top-n",         type=int, default=10,
                        help="Top-N grasps per object to test (default: 10)")
    parser.add_argument("--obj-ids",       type=int, nargs='*', default=None,
                        help="Object IDs to include (default: all found in scene annotation)")
    parser.add_argument("--executor",      choices=["descend", "teleport"], default="descend",
                        help="Grasp execution strategy (default: descend)")
    parser.add_argument("--video",         action="store_true",
                        help="Record and save videos (slower)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip individual-object experiments")
    parser.add_argument("--no-combined",   action="store_true",
                        help="Skip combined-scene experiment")
    parser.add_argument("--overlay-corner",
                        choices=["top_left", "top_right", "none"],
                        default="top_left",
                        help="Where to draw the 'obj_XXX grasp #N' overlay on video frames "
                             "(default: top_left; 'none' disables the overlay).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    GRASPS_NPY  = args.grasps_npy
    CAMERA_EXTR = args.camera_extr
    CAMERA_POSE = args.camera_pose

    executor = Executors.descend if args.executor == "descend" else Executors.teleport

    scene = Scene(
        scene_dir=args.scene_dir,
        model_dir=args.model_dir,
        hand_assets=args.hand_assets,
        camera=args.camera,
    )

    obj_ids = args.obj_ids if args.obj_ids is not None else scene.obj_indexes_list
    logger.info(f"Objects in scene: {obj_ids}")

    gg = GraspGroup(np.load(GRASPS_NPY))
    logger.info(f"Total grasps loaded: {len(gg)}")
    logger.info(f"Top-N per object: {args.top_n}  |  executor: {args.executor}  |  video: {args.video}")

    all_records = []

    if not args.no_individual:
        all_records += run_individual(scene, gg, obj_ids, args.top_n,
                                      output_dir, executor, args.video,
                                      overlay_corner=args.overlay_corner)

    if not args.no_combined:
        all_records += run_combined(scene, gg, obj_ids, args.top_n,
                                    output_dir, executor, args.video,
                                    overlay_corner=args.overlay_corner)

    if not all_records:
        logger.warning("No results collected. Check object IDs and grasp file.")
        return

    csv_path = output_dir / "results.csv"
    fieldnames = list(all_records[0].keys())
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    logger.info(f"Results CSV  -> {csv_path}")

    summary = _summarize(all_records)
    json_path = output_dir / "summary.json"
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary JSON -> {json_path}")

    _print_summary(summary)


if __name__ == "__main__":
    main()
