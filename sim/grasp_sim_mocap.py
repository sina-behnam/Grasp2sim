import os
from turtle import width
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.append("/home/sbehnam/Project/grasp2sim")

import numpy as np
import mujoco
import mediapy as media
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from graspnetAPI import GraspGroup, Grasp
import argparse

from loguru import logger
from utils.poses import gg_filter_by_object_id, gg_filter_by_width, gg_filter_by_orthogonal_approach
from sim.logger.sim_logger import SimLogger
from sim.logger.grasp_debugger import GraspDebugger, Phase

# Config
SCENE_XML   = "/home/sbehnam/Project/grasp2sim/scenes/scene_0000_mocap_simple.xml"
GRASPS_NPY  = "/home/sbehnam/Project/data/scenes/scene_0000/grasp_group_mine.npy"
CAMERA_EXTR = "/home/sbehnam/Project/data/scenes/scene_0000/kinect/cam0_wrt_table.npy"
CAMERA_POSE = "/home/sbehnam/Project/data/scenes/scene_0000/kinect/camera_poses.npy"

LIFT_HEIGHT    = 0.08
CAPTURE_EVERY  = 15

FINGER_BASE_Z      = 0.0584
FINGERTIP_PAD_Z    = 0.0445
FINGERTIP_PAD_HALF = 0.0085
FINGERTIP_OFFSET   = FINGER_BASE_Z + FINGERTIP_PAD_Z + FINGERTIP_PAD_HALF

HOME_POS  = np.array([0.0, 0.0, 0.6])
HOME_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz


def _wxyz_to_xyzw(q):  return np.array([q[1], q[2], q[3], q[0]])
def _xyzw_to_wxyz(q):  return np.array([q[3], q[0], q[1], q[2]])

logger.info("Logger initialized.")

class GraspHandMocap:
    """
    Simulated mocap-weld hand for grasp execution and evaluation in MuJoCo.

    Difference from GraspHand: the hand body carries a freejoint and is
    coupled to a mocap body (hand_target) via a weld equality constraint.
    Python drives sim.mocap_pos / sim.mocap_quat; constraint forces pull the
    physical hand to the target each timestep, so contacts with objects are
    fully dynamic.
    """

    def __init__(self, scene_xml=SCENE_XML,camera_extr=CAMERA_EXTR, camera_pose=CAMERA_POSE,
                 render='human', camera=None, debug=False, debug_log_every=1, seed=42,
                 grasp_debug=False, grasp_debug_mu=1.0):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.sim   = mujoco.MjData(self.model)

        np.random.seed(seed)

        cam_2_table      = np.load(camera_extr)
        camera_poses     = np.load(camera_pose)
        self.T_CAM2TABLE = cam_2_table @ camera_poses[0]

        self.OBJ_BODY_NAMES = [f"obj_{i:03d}" for i in range(100)]
        self.obj_ids = []
        self.obj_names = []
        for n in self.OBJ_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid != -1:
                self.obj_ids.append(bid)
                self.obj_names.append(n)

        # Mocap target — Python drives this
        target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_target")
        assert target_bid != -1, "hand_target mocap body not found"
        self.mocap_idx = self.model.body_mocapid[target_bid]
        assert self.mocap_idx != -1, "hand_target is not a mocap body"

        # Freejoint on the hand body — needed for direct teleportation
        self.freejoint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_freejoint")
        assert self.freejoint_id != -1, "hand_freejoint not found"

        # Renderer
        if render != 'human':
            self.renderer = None
        else:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.frames = []
            if camera is None:
                self.cam = mujoco.MjvCamera()
                self.cam.azimuth   = 160
                self.cam.elevation = -30
                self.cam.distance  = 1.0
                self.cam.lookat[:] = [0.0, -0.15, 0.0]
            else:
                self.cam = camera

        self.debug = debug
        self.logger = SimLogger(self.model, self.sim,
                                body_names=self.obj_names + ["hand"],
                                log_every=debug_log_every) if debug else None

        self.grasp_debugger = GraspDebugger(self.model, self.sim, mu_estimate=grasp_debug_mu) \
            if grasp_debug else None

    def _dlog(self):
        if self.logger is not None:
            self.logger.log()
        if self.grasp_debugger is not None:
            self.grasp_debugger.log_step()

    def mark_phase(self, name):
        """No-op when grasp_debugger is disabled."""
        if self.grasp_debugger is not None:
            self.grasp_debugger.mark_phase(name)

    # Geometry
    def grasp_to_world(self, g : Grasp):
        t_w = self.T_CAM2TABLE[:3, :3] @ g.translation + self.T_CAM2TABLE[:3, 3]
        approach_w = self.T_CAM2TABLE[:3, :3] @ g.rotation_matrix[:, 0]
        return t_w - approach_w * (FINGERTIP_OFFSET - g.depth)

    def to_mujoco_quat(self, R_cam):
        R_w      = self.T_CAM2TABLE[:3, :3] @ R_cam
        approach = R_w[:, 0]
        binormal = R_w[:, 1]
        minor    = R_w[:, 2]
        R_hand   = np.column_stack([-minor, binormal, approach])
        q_xyzw = Rot.from_matrix(R_hand).as_quat()
        return _xyzw_to_wxyz(q_xyzw)

    # Mocap hand control
    def set_hand_pose(self, pos, quat_wxyz):
        """
        Teleport hand to pose by setting both the mocap target and the
        freejoint qpos directly, avoiding weld-constraint transients.
        """
        # Move mocap target
        self.sim.mocap_pos[self.mocap_idx]  = pos
        self.sim.mocap_quat[self.mocap_idx] = quat_wxyz  # MuJoCo mocap_quat is wxyz

        # Also snap the freejoint so the hand starts at the same position
        # (avoids large constraint impulse on the next step)
        qadr = self.model.jnt_qposadr[self.freejoint_id]
        self.sim.qpos[qadr:qadr + 3] = pos
        self.sim.qpos[qadr + 3:qadr + 7] = quat_wxyz  # freejoint qpos is [x,y,z, w,x,y,z]
        mujoco.mj_forward(self.model, self.sim)

    def get_hand_pose(self):
        """Return the current mocap target pose (what we commanded)."""
        return (self.sim.mocap_pos[self.mocap_idx].copy(),
                self.sim.mocap_quat[self.mocap_idx].copy())

    def move_hand(self, target_pos, target_quat, n_steps,
                  record=False, substeps=5, settle_steps=40, ease='cosine'):
        """
        Smoothly drive the mocap target from current pose to target.
        The weld constraint pulls the physical hand along each substep.
        """
        start_pos, start_quat = self.get_hand_pose()
        q_start  = _wxyz_to_xyzw(start_quat)
        q_target = _wxyz_to_xyzw(target_quat)
        slerp_fn = Slerp([0.0, 1.0],
                         Rot.concatenate([Rot.from_quat(q_start), Rot.from_quat(q_target)]))

        for i in range(n_steps):
            s = (i + 1) / n_steps
            t = 0.5 * (1.0 - np.cos(np.pi * s)) if ease == 'cosine' else s
            pos  = (1 - t) * start_pos + t * target_pos
            quat = _xyzw_to_wxyz(slerp_fn([t]).as_quat()[0])
            # Only update mocap target here — do NOT snap freejoint mid-motion
            self.sim.mocap_pos[self.mocap_idx]  = pos
            self.sim.mocap_quat[self.mocap_idx] = quat
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.sim)
                self._dlog()
            if record and i % CAPTURE_EVERY == 0:
                self.capture()
        for j in range(settle_steps):
            mujoco.mj_step(self.model, self.sim)
            self._dlog()
            if record and j % CAPTURE_EVERY == 0:
                self.capture()

    # Gripper
    def open_gripper(self, width=0.08):
        width = np.clip(width, 0.0, 0.08)
        q     = 0.5 * width                # per-finger slide
        self.sim.ctrl[0] = (q / 0.04) * 255


    def close_gripper(self):
        self.sim.ctrl[0] = 0

    # Stepping / rendering
    def step(self, n, record=False):
        for i in range(n):
            mujoco.mj_step(self.model, self.sim)
            self._dlog()
            if record and i % CAPTURE_EVERY == 0:
                self.capture()

    def capture(self):
        if self.renderer is None:
            return
        self.renderer.update_scene(self.sim, self.cam)
        self.frames.append(self.renderer.render().copy())

    # Reset
    def reset_scene(self):
        mujoco.mj_resetData(self.model, self.sim)
        if self.logger is not None:
            self.logger.reset()
        self.set_hand_pose(HOME_POS, HOME_QUAT)
        self.open_gripper(0.08)

    # Single-grasp evaluation
    def run_grasp(self, g : Grasp, executor : callable, grasp_index=0):
        t_w        = self.grasp_to_world(g)
        quat       = self.to_mujoco_quat(g.rotation_matrix)
        approach_w = self.T_CAM2TABLE[:3, :3] @ g.rotation_matrix[:, 0]
        binormal_w = self.T_CAM2TABLE[:3, :3] @ g.rotation_matrix[:, 1]

        self.reset_scene()
        # self.open_gripper(min(0.08, g.width + 0.015))
        self.open_gripper(0.08)
        self.step(30)
        self.capture()

        z0 = np.array([self.sim.xpos[oid][2] for oid in self.obj_ids])

        if self.grasp_debugger is not None:
            self.grasp_debugger.begin_grasp(grasp_index, g.score, g.object_id, t_w, g.width,
                                            approach_baseline=approach_w,
                                            binormal_baseline=binormal_w)

        executor(self, t_w, quat, g.width, approach_w=approach_w)

        z1 = np.array([self.sim.xpos[oid][2] for oid in self.obj_ids])
        lift = z1 - z0
        lifted = [(self.obj_names[i], float(lift[i]))
                  for i in range(len(self.obj_ids)) if lift[i] >= LIFT_HEIGHT]

        if self.grasp_debugger is not None:
            self.grasp_debugger.end_grasp(success=len(lifted) > 0, lifted=lifted)

        return t_w, lifted

    def save_video(self, path, fps=8):
        media.write_video(path, self.frames, fps=fps)
        print(f"Saved {path}")

class Executors:
    """Grasp execution strategies for the mocap-weld hand."""

    @staticmethod
    def teleport(exe: GraspHandMocap, t_w, quat, width, approach_w=None):
        exe.mark_phase(Phase.APPROACH)
        exe.set_hand_pose(t_w, quat)
        exe.capture()
        exe.step(30, record=True)

        exe.mark_phase(Phase.CLOSE)
        exe.close_gripper()
        exe.step(200, record=True)

        exe.mark_phase(Phase.RETREAT)
        if approach_w is not None:
            exe.move_hand(t_w - 0.10 * approach_w, quat,
                          n_steps=120, record=True)

        exe.mark_phase(Phase.LIFT)
        lift_pos = exe.get_hand_pose()[0] + np.array([0.0, 0.0, 0.30])
        exe.move_hand(lift_pos, quat, n_steps=200, record=True)
        exe.mark_phase(Phase.DONE)

    @staticmethod
    def descend(exe: GraspHandMocap, t_w, quat, width, approach_w=None, standoff=0.12):
        if approach_w is None:
            approach_w = np.array([0.0, 0.0, -1.0])
        pre_t_w = t_w - standoff * approach_w

        exe.mark_phase(Phase.APPROACH)
        exe.move_hand(pre_t_w, quat, n_steps=200, record=True)
        exe.open_gripper(width + 0.02)
        exe.step(50, record=True)
        exe.move_hand(t_w, quat, n_steps=100, record=True, substeps=8)

        exe.mark_phase(Phase.CLOSE)
        exe.close_gripper()
        exe.step(200, record=True)

        exe.mark_phase(Phase.RETREAT)
        exe.move_hand(pre_t_w, quat, n_steps=200, record=True)

        exe.mark_phase(Phase.LIFT)
        exe.move_hand(pre_t_w + np.array([0.0, 0.0, 0.25]), quat, n_steps=200, record=True)
        exe.mark_phase(Phase.DONE)

def main():

    global GRASPS_NPY, CAMERA_EXTR, CAMERA_POSE, SCENE_XML

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-xml",       default=SCENE_XML)
    parser.add_argument("--grasps-npy",      default=GRASPS_NPY)
    parser.add_argument("--camera-extr",     default=CAMERA_EXTR)
    parser.add_argument("--camera-pose",     default=CAMERA_POSE)
    parser.add_argument("--object-id",      type=int, default=-1)
    parser.add_argument("--render",          default="human")
    parser.add_argument("--debug",           action="store_true")
    parser.add_argument("--debug-log-every", type=int, default=1)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--top",             type=int, default=20)
    parser.add_argument("--output",          default="test-mocap2.mp4")
    parser.add_argument("--grasp-debug",     action="store_true",
                        help="Enable per-grasp contact/phase instrumentation.")
    parser.add_argument("--grasp-debug-out", default="grasp_debug",
                        help="Directory for grasp-debug artefacts (records, plots, csv).")
    parser.add_argument("--grasp-debug-mu",  type=float, default=1.0,
                        help="Estimated friction coefficient for friction-cone plot annotation.")
    args = parser.parse_args()

    GRASPS_NPY  = args.grasps_npy
    CAMERA_EXTR = args.camera_extr
    CAMERA_POSE = args.camera_pose
    SCENE_XML   = args.scene_xml

    exe = GraspHandMocap(
        scene_xml=SCENE_XML,
        camera_extr=CAMERA_EXTR,
        camera_pose=CAMERA_POSE,
        render=args.render,
        debug=args.debug,
        debug_log_every=args.debug_log_every,
        seed=args.seed,
        grasp_debug=args.grasp_debug,
        grasp_debug_mu=args.grasp_debug_mu,
    )
    executor   = Executors.descend
    video_path = args.output

    gg = GraspGroup(np.load(GRASPS_NPY))

    if args.object_id != -1:
        test_grasps = gg_filter_by_object_id(gg, object_id=args.object_id)
    else:
        test_grasps = gg

    num_pre_width = len(test_grasps)

    test_grasps = gg_filter_by_width(test_grasps, width_threshold=0.08)

    num_post_width = len(test_grasps)

    if num_pre_width != num_post_width:
        logger.warning(f"Filtered out {num_pre_width - num_post_width} grasps due to width > 0.08")

    test_grasps = gg_filter_by_orthogonal_approach(test_grasps, 
                                        orthogonal_threshold=np.cos(np.radians(25)),
                                        table_to_cam=exe.T_CAM2TABLE)
    
    test_grasps.sort_by_score()

    sampled_grasps = test_grasps[:args.top]

    if len(sampled_grasps) == 0:
        logger.error("No grasps to test after filtering. Exiting. \n [Hint] Check the object_id and width_threshold parameters.")
        return

    logger.info(f"Testing {len(sampled_grasps)} grasps | executor = {executor.__name__}")

    results = []
    for rank in range(len(sampled_grasps)):
        g = sampled_grasps[rank]
        t_w, lifted = exe.run_grasp(g, executor, grasp_index=rank)
        success = len(lifted) > 0
        results.append((g.score, success, lifted))
        logger.info(f"[{rank+1}/{len(sampled_grasps)}] score={g.score:.3f} object_id={g.object_id} "
              f"t_w={np.round(t_w, 3)}  "
              f"{'SUCCESS' if success else 'FAIL'}  lifted={lifted}")

    exe.save_video(video_path, fps=5)
    n_ok = sum(1 for _, s, _ in results if s)
    logger.info(f"Success rate: {n_ok}/{len(results)} = {100*n_ok/len(results):.1f}%")

    if exe.grasp_debugger is not None:
        exe.grasp_debugger.print_summary(log=logger.info)
        exe.grasp_debugger.dump(args.grasp_debug_out)
        logger.info(f"Grasp-debug artefacts saved -> {args.grasp_debug_out}")


if __name__ == "__main__":
    main()
