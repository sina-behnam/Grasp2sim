import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.append("/home/sbehnam/Project/grasp2sim")

import numpy as np
import mujoco
import mediapy as media
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from graspnetAPI import GraspGroup

from poses.publisher import gg_filter_by_object_id

from sim_logger import SimLogger

# Config
SCENE_XML   = "/home/sbehnam/Project/grasp2sim/scenes/scene_5-2_0000.xml"
# GRASPS_NPY  = "/home/sbehnam/Project/data/scenes/scene_0000/grasp_group_mine.npy"
GRASPS_NPY  = "/home/sbehnam/Project/data/scenes/scene_0000/some_banana_grasp.npy"  # smaller set for quick testing
CAMERA_EXTR = "/home/sbehnam/Project/data/scenes/scene_0000/kinect/cam0_wrt_table.npy"
CAMERA_POSE = "/home/sbehnam/Project/data/scenes/scene_0000/kinect/camera_poses.npy"

LIFT_HEIGHT    = 0.08
CAPTURE_EVERY  = 15

# Panda fingertip geometry
FINGER_BASE_Z      = 0.0584
FINGERTIP_PAD_Z    = 0.0445
FINGERTIP_PAD_HALF = 0.0085
FINGERTIP_OFFSET   = FINGER_BASE_Z + FINGERTIP_PAD_Z + FINGERTIP_PAD_HALF
# FINGERTIP_OFFSET = FINGER_BASE_Z + FINGERTIP_PAD_Z

HOME_POS  = np.array([0.0, 0.0, 0.6])
HOME_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz


def _wxyz_to_xyzw(q):  return np.array([q[1], q[2], q[3], q[0]])
def _xyzw_to_wxyz(q):  return np.array([q[3], q[0], q[1], q[2]])


class GraspHand:
    """
    Simulated kinematic hand for grasp execution and evaluation in MuJoCo.
    """

    def __init__(self, scene_xml=SCENE_XML, grasps_npy=GRASPS_NPY,
                 camera_extr=CAMERA_EXTR, camera_pose=CAMERA_POSE ,render='human', camera=None,
                 debug=False, debug_log_every=1):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.sim   = mujoco.MjData(self.model)

        cam_2_table      = np.load(camera_extr)
        camera_poses     = np.load(camera_pose)
        self.T_CAM2TABLE = cam_2_table @ camera_poses[0]  # Always use the first camera pose !!!
        self.gg          = GraspGroup(np.load(grasps_npy))

        self.OBJ_BODY_NAMES = [f"obj_{i:03d}" for i in range(100)]
        self.obj_ids = []
        self.obj_names = []
        for n in self.OBJ_BODY_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid != -1:
                self.obj_ids.append(bid)
                self.obj_names.append(n)

        # Hand: kinematic base — we drive model.body_pos/body_quat directly
        self.hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        assert self.hand_bid != -1, "hand body not found"

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

        # Debug logging (off by default)
        self.debug = debug
        self.logger = SimLogger(self.model, self.sim,
                                body_names=self.obj_names + ["hand"],
                                log_every=debug_log_every) if debug else None

    def _dlog(self):
        if self.logger is not None:
            self.logger.log()

    # Geometry
    def grasp_to_world(self, g):
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

    # Kinematic hand control (Pattern C core)
    def set_hand_pose(self, pos, quat_wxyz):
        """Instantaneously place the hand base. No physics on the base."""
        self.model.body_pos[self.hand_bid]  = pos
        self.model.body_quat[self.hand_bid] = quat_wxyz
        mujoco.mj_forward(self.model, self.sim)

    def get_hand_pose(self):
        return (self.model.body_pos[self.hand_bid].copy(),
                self.model.body_quat[self.hand_bid].copy())

    def move_hand(self, target_pos, target_quat, n_steps,
                  record=False, substeps=5, settle_steps=40, ease='cosine'):
        """
        Smoothly command the kinematic hand from current pose to target.
        Uses cosine ease-in/out for zero start/end velocity; slerp for rotation.
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
            self.set_hand_pose(pos, quat)
            for _ in range(substeps):
                mujoco.mj_step(self.model, self.sim)
                self._dlog()
            if record and i % CAPTURE_EVERY == 0:
                self.capture()
        # Settle: hold target pose, let fingers/objects reach equilibrium
        for j in range(settle_steps):
            mujoco.mj_step(self.model, self.sim)
            self._dlog()
            if record and j % CAPTURE_EVERY == 0:
                self.capture()

    # Gripper
    def open_gripper(self, width=0.08):
        self.sim.ctrl[0] = (np.clip(width, 0.0, 0.08) / 0.08) * 255

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

    # Reset at the start of every grasp trial
    def reset_scene(self):
        mujoco.mj_resetData(self.model, self.sim)
        if self.logger is not None:
            self.logger.reset()
        self.set_hand_pose(HOME_POS, HOME_QUAT)
        self.open_gripper(0.08)
        # For stability, step the sim for a few frames before starting the grasp trial.
        # self.step(300)

    # Single-grasp evaluation
    def run_grasp(self, g, executor, mode='slerp'):
        t_w        = self.grasp_to_world(g)
        quat       = self.to_mujoco_quat(g.rotation_matrix)
        approach_w = self.T_CAM2TABLE[:3, :3] @ g.rotation_matrix[:, 0]

        self.reset_scene()
        self.open_gripper(min(0.08, g.width + 0.015))   # small slack
        self.step(30)
        self.capture()

        z0 = np.array([self.sim.xpos[oid][2] for oid in self.obj_ids])

        executor(self, t_w, quat, mode=mode, approach_w=approach_w)

        z1 = np.array([self.sim.xpos[oid][2] for oid in self.obj_ids])
        lift = z1 - z0
        lifted = [(self.obj_names[i], float(lift[i]))
                  for i in range(len(self.obj_ids)) if lift[i] >= LIFT_HEIGHT]
        return t_w, lifted

    def save_video(self, path, fps=8):
        media.write_video(path, self.frames, fps=fps)
        print(f"Saved {path}")


class Executors:
    """Grasp execution strategies for the kinematic hand."""

    @staticmethod
    def teleport(exe: GraspHand, t_w, quat, mode='slerp', approach_w=None):
        """
        Pattern-C natural: instantaneously snap hand to grasp pose, close, lift.
        Safe because the hand base is kinematic — no impulses from the snap.
        """
        exe.set_hand_pose(t_w, quat)
        exe.capture()
        exe.step(30, record=True)           # let fingers re-settle around pose
        exe.close_gripper()
        exe.step(200, record=True)          # finger closure on object
        # Retreat along approach axis, then lift vertically
        if approach_w is not None:
            exe.move_hand(t_w - 0.10 * approach_w, quat,
                          n_steps=120, record=True)
        lift_pos = exe.get_hand_pose()[0] + np.array([0.0, 0.0, 0.30])
        exe.move_hand(lift_pos, quat, n_steps=200, record=True)

    @staticmethod
    def descend(exe: GraspHand, t_w, quat, mode='slerp',
                approach_w=None, standoff=0.12):
        """
        Pre-grasp -> axial approach -> close -> retreat -> lift.
        Standoff 12 cm > FINGERTIP_OFFSET (10.3 cm) ensures clearance.
        """
        if approach_w is None:
            approach_w = np.array([0.0, 0.0, -1.0])
        pre_t_w = t_w - standoff * approach_w

        # Phase A: free-space move to pre-grasp (rotation + translation together)
        exe.move_hand(pre_t_w, quat, n_steps=200, record=True)
        # Phase B: axial-only approach (same quat, pure translation along approach)
        exe.move_hand(t_w, quat, n_steps=200, record=True, substeps=8)
        # Close around object
        exe.close_gripper()
        exe.step(200, record=True)
        # Retreat along approach axis first
        exe.move_hand(pre_t_w, quat, n_steps=150, record=True)
        # Then lift vertically
        exe.move_hand(pre_t_w + np.array([0.0, 0.0, 0.25]), quat,
                      n_steps=200, record=True)


def main():
    exe = GraspHand()
    executor   = Executors.descend
    video_path = "test-patternC.mp4"

    exe.gg.sort_by_score()
    # Pick grasps for a specific object (object_id is last column of grasp array)
    one_obj_grasps = gg_filter_by_object_id(exe.gg, object_id=5)
    test_grasps = one_obj_grasps.random_sample(min(50, len(one_obj_grasps)))

    print(f"Testing {len(test_grasps)} grasps | executor = {executor.__name__}")

    results = []
    for rank in range(len(test_grasps)):
        g = test_grasps[rank]
        t_w, lifted = exe.run_grasp(g, executor, mode='slerp')
        success = len(lifted) > 0
        results.append((g.score, success, lifted))
        print(f"[{rank+1}/{len(test_grasps)}] score={g.score:.3f} "
              f"t_w={np.round(t_w, 3)}  "
              f"{'SUCCESS' if success else 'FAIL'}  lifted={lifted}")

    exe.save_video(video_path)
    n_ok = sum(1 for _, s, _ in results if s)
    print(f"Success rate: {n_ok}/{len(results)} = {100*n_ok/len(results):.1f}%")


if __name__ == "__main__":
    main()