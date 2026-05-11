#!/usr/bin/env python3
"""
Grasp experiment runner: per-object isolation vs. combined-scene success rates.

For each object found in the scene annotation:
  1. Individual  — scene contains only that object; top-N grasps are tested.
  2. Combined    — scene contains all objects; top-N grasps per object are tested
                   (success = the *target* object was lifted ≥ LIFT_HEIGHT).

Outputs written to <output_dir>/:
  results.csv               one row per grasp trial
  summary.json              success rates keyed by "experiment/obj_XXX"
  obj_XXX_individual.mp4    per-object video  (individual mode, if --video)
  combined.mp4              all grasps video  (combined mode,  if --video)
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

from sim.grasp_sim_mocap import GraspHandMocap, Executors
from scenes.grasp2scene_mocap import Scene
from utils.poses import gg_filter_by_object_id, gg_filter_by_width, gg_filter_by_orthogonal_approach

SCENE_DIR   = "/home/sbehnam/Project/data/scenes/scene_0000"
MODEL_DIR   = "/home/sbehnam/models"
HAND_ASSETS = "/home/sbehnam/Project/grasp2sim/franka_emika_panda/assets"
CAMERA      = "kinect"
GRASPS_NPY  = f"{SCENE_DIR}/grasp_group_mine.npy"
CAMERA_EXTR = f"{SCENE_DIR}/kinect/cam0_wrt_table.npy"
CAMERA_POSE = f"{SCENE_DIR}/kinect/camera_poses.npy"
OUTPUT_DIR  = "/home/sbehnam/Project/grasp2sim/experiment_results"

def _prepare_grasps(gg: GraspGroup, obj_id: int, top_n: int) -> GraspGroup:
    """Filter by object ID and width, sort by score, return top-N."""
    g = gg_filter_by_object_id(gg, object_id=obj_id)
    g = gg_filter_by_width(g, width_threshold=0.08)
    g = gg_filter_by_orthogonal_approach(g,
                                        orthogonal_threshold=np.cos(np.radians(25)),
                                        table_to_cam=np.load(CAMERA_EXTR))
    g.sort_by_score()
    return g[:top_n]

def _make_sim(xml_path: str, camera_extr: str,
              camera_pose: str, video: bool) -> GraspHandMocap:
    return GraspHandMocap(
        scene_xml=xml_path,
        camera_extr=camera_extr,
        camera_pose=camera_pose,
        render='human' if video else 'off',
    )


def _run_grasps(sim: GraspHandMocap, sampled: GraspGroup,
                executor, target_obj_name: str | None) -> list:
    """
    Run every grasp in `sampled` through `sim.run_grasp`.

    target_obj_name: if given, success = that object was lifted (combined mode).
                     if None, success = anything was lifted (individual mode).
    """
    records = []
    for rank in range(len(sampled)):
        g = sampled[rank]
        t_w, lifted = sim.run_grasp(g, executor)
        if target_obj_name is None:
            success = len(lifted) > 0
        else:
            success = any(name == target_obj_name for name, _ in lifted)
        records.append({
            'rank': rank + 1,
            'score': float(g.score),
            'width': float(g.width),
            'success': success,
            'lifted': [(n, round(h, 4)) for n, h in lifted],
        })
        status = 'SUCCESS' if success else 'FAIL'
        logger.info(f"    [{rank+1}/{len(sampled)}] score={g.score:.3f}  {status}  lifted={lifted}")
    return records


# ── individual experiments ─────────────────────────────────────────────────────

def run_individual(scene: Scene, gg: GraspGroup, obj_ids: list,
                   top_n: int, output_dir: Path,
                   executor, video: bool) -> list:
    """One scene per object; returns list of result dicts."""
    all_records = []
    for obj_id in obj_ids:
        logger.info(f"=== INDIVIDUAL  obj_{obj_id:03d} ===")

        sampled = _prepare_grasps(gg, obj_id, top_n)
        if len(sampled) == 0:
            logger.warning(f"No grasps for obj_{obj_id:03d} — skipping.")
            continue

        # Write a temporary XML with only this object
        with tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                         mode='w', dir=output_dir) as fh:
            xml_path = fh.name
        try:
            scene.save_xml(xml_path, obj_indexes=[obj_id], coacd=True, strength='original')
            sim = _make_sim(xml_path, str(CAMERA_EXTR),
                            str(CAMERA_POSE), video)

            logger.info(f"  Testing {len(sampled)} grasps")
            records = _run_grasps(sim, sampled, executor, target_obj_name=None)

            if video:
                vpath = output_dir / f"obj_{obj_id:03d}_individual.mp4"
                sim.save_video(str(vpath), fps=5)
                logger.info(f"  Video → {vpath}")

        finally:
            os.unlink(xml_path) # clean up temp file

        n_ok = sum(1 for r in records if r['success'])
        logger.info(f"  obj_{obj_id:03d} individual: {n_ok}/{len(records)}")

        for r in records:
            all_records.append({
                'experiment': 'individual',
                'obj_id': obj_id,
                **r,
            })

    return all_records


# ── combined experiment ────────────────────────────────────────────────────────

def run_combined(scene: Scene, gg: GraspGroup, obj_ids: list,
                 top_n: int, output_dir: Path,
                 executor, video: bool) -> list:
    """All objects in one scene; test top-N grasps per object."""
    logger.info("=== COMBINED  (all objects together) ===")

    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                     mode='w', dir=output_dir) as fh:
        xml_path = fh.name
    try:
        scene.save_xml(xml_path, obj_indexes=None, coacd=True, strength='original')  # full scene
        sim = _make_sim(xml_path, str(CAMERA_EXTR),
                        str(CAMERA_POSE), video)

        all_records = []
        for obj_id in obj_ids:
            sampled = _prepare_grasps(gg, obj_id, top_n)
            if len(sampled) == 0:
                logger.warning(f"No grasps for obj_{obj_id:03d} in combined — skipping.")
                continue

            logger.info(f"  obj_{obj_id:03d}: {len(sampled)} grasps")
            target_name = f"obj_{obj_id:03d}"
            records = _run_grasps(sim, sampled, executor, target_obj_name=target_name)

            n_ok = sum(1 for r in records if r['success'])
            logger.info(f"  obj_{obj_id:03d} combined:    {n_ok}/{len(records)}")

            for r in records:
                all_records.append({
                    'experiment': 'combined',
                    'obj_id': obj_id,
                    **r,
                })

        if video:
            vpath = output_dir / "combined.mp4"
            sim.save_video(str(vpath), fps=5)
            logger.info(f"  Video → {vpath}")

    finally:
        os.unlink(xml_path)

    return all_records


# ── summary ────────────────────────────────────────────────────────────────────

def _summarize(records: list) -> dict:
    counts = defaultdict(lambda: {'success': 0, 'total': 0})
    for r in records:
        key = f"{r['experiment']}/obj_{r['obj_id']:03d}"
        counts[key]['total'] += 1
        if r['success']:
            counts[key]['success'] += 1
    return {
        k: {**v, 'rate': round(v['success'] / v['total'], 3) if v['total'] else 0.0}
        for k, v in sorted(counts.items())
    }


def _print_summary(summary: dict):
    logger.info("")
    logger.info("┌─────────────────────────────────────────┬──────────┬────────┐")
    logger.info("│ Experiment / Object                     │  Result  │  Rate  │")
    logger.info("├─────────────────────────────────────────┼──────────┼────────┤")
    for key, s in summary.items():
        logger.info(f"│ {key:<39} │ {s['success']:>3}/{s['total']:<3}  │ {100*s['rate']:5.1f}% │")
    logger.info("└─────────────────────────────────────────┴──────────┴────────┘")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # Global paths (allow CLI overrides)
    global GRASPS_NPY, CAMERA_EXTR, CAMERA_POSE

    parser = argparse.ArgumentParser(
        description="Run grasp experiments: per-object isolation vs. combined scene.")
    parser.add_argument("--scene-dir",    default=SCENE_DIR)
    parser.add_argument("--model-dir",    default=MODEL_DIR)
    parser.add_argument("--hand-assets",  default=HAND_ASSETS)
    parser.add_argument("--camera",       default=CAMERA)
    parser.add_argument("--grasps-npy",   default=GRASPS_NPY)
    parser.add_argument("--camera-extr",  default=CAMERA_EXTR)
    parser.add_argument("--camera-pose",  default=CAMERA_POSE)
    parser.add_argument("--output-dir",   default=OUTPUT_DIR)
    parser.add_argument("--top-n",        type=int, default=10,
                        help="Top-N grasps per object to test (default: 10)")
    parser.add_argument("--obj-ids",      type=int, nargs='*', default=None,
                        help="Object IDs to include (default: all found in scene annotation)")
    parser.add_argument("--executor",     choices=["descend", "teleport"], default="descend",
                        help="Grasp execution strategy (default: descend)")
    parser.add_argument("--video",        action="store_true",
                        help="Record and save videos (slower)")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip individual-object experiments")
    parser.add_argument("--no-combined",   action="store_true",
                        help="Skip combined-scene experiment")
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
                                      output_dir, executor, args.video)

    if not args.no_combined:
        all_records += run_combined(scene, gg, obj_ids, args.top_n,
                                    output_dir, executor, args.video)

    if not all_records:
        logger.warning("No results collected. Check object IDs and grasp file.")
        return

    # Save CSV
    csv_path = output_dir / "results.csv"
    fieldnames = list(all_records[0].keys())
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    logger.info(f"Results CSV  → {csv_path}")

    # Save summary JSON
    summary = _summarize(all_records)
    json_path = output_dir / "summary.json"
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary JSON → {json_path}")

    _print_summary(summary)


if __name__ == "__main__":
    main()
