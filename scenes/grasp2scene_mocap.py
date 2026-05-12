import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import parse_posevector

import argparse

GRASPNET_SCENE_ROOT = '/home/sbehnam/Project/data/scenes/scene_0000'
MODEL_DIR   = '/home/sbehnam/models'
HAND_ASSETS = '/home/sbehnam/Project/grasp2sim/franka_emika_panda/assets'
CAMERA      = 'kinect'
OUTPUT_XML  = '/home/sbehnam/Project/grasp2sim/outputs/scenes/scene_0000_mocap.xml'


class Scene:
    def __init__(self, 
                scene_dir   = GRASPNET_SCENE_ROOT,
                model_dir   = MODEL_DIR,
                hand_assets = HAND_ASSETS,
                camera      = CAMERA):
        self.camera      = camera
        self.scene_dir   = os.path.join(scene_dir, camera)
        self.model_dir   = model_dir
        self.hand_assets = hand_assets
        self.camera_poses = np.load(os.path.join(self.scene_dir, 'camera_poses.npy'))
        self.align_mat    = np.load(os.path.join(self.scene_dir, 'cam0_wrt_table.npy'))
        xml_reader        = xmlReader(os.path.join(self.scene_dir, 'annotations', '0000.xml'))
        self.posevectors  = xml_reader.getposevectorlist()

    def _gripper_strength(self, strength : str = 'original'):
        if strength == 'original':
            return '''
<general class="panda" name="actuator8" tendon="split" forcerange="-100 100"
                      ctrlrange="0 255" gainprm="0.01568627451 0 0" biasprm="0 -100 -10"/>
'''
        elif strength == 'strong':
            return '''
<general class="panda" name="actuator8" tendon="split" forcerange="-300 300" ctrlrange="0 255" gainprm="0.04 0 0" biasprm="0 -300 -30"/>
'''
        else:
            raise ValueError(f'Invalid strength: {strength}')

    @property
    def camera_pose(self):
        return self.align_mat @ self.camera_poses[0]

    @staticmethod
    def mat_to_quat_wxyz(T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        return q[[3, 0, 1, 2]]

    @property
    def obj_indexes_list(self):
        return [parse_posevector(pv)[0] for pv in self.posevectors]

    def _coacd_parts(self, obj_str):
        return sorted(glob.glob(os.path.join(
            self.model_dir, obj_str, 'coacd_part_*.stl')))

    def _build_obj_lines(self, obj_indexes=None, coacd=False):
        if isinstance(obj_indexes, int):
            obj_indexes = [obj_indexes]
        asset_lines, body_lines = [], []

        OBJ_FRICTION = '2.0 0.05 0.001'
        OBJ_SOLIMP   = '0.99 0.999 0.0001'
        OBJ_SOLREF   = '0.005 1'
        OBJ_CONDIM   = '4'
        OBJ_DENSITY  = '500'

        for pv in self.posevectors:
            obj_idx, obj_pose_cam = parse_posevector(pv)
            if obj_indexes is not None and obj_idx not in obj_indexes:
                continue
            T       = self.camera_pose @ obj_pose_cam
            obj_str = f'{obj_idx:03d}'
            t       = T[:3, 3]
            q       = self.mat_to_quat_wxyz(T)

            part_files = self._coacd_parts(obj_str) if coacd else []

            if coacd and part_files:
                asset_lines.append(
                    f'    <mesh name="vis_obj_{obj_str}" file="{self.model_dir}/{obj_str}/nontextured.stl"/>'
                )
                geom_lines = [
                    f'                  <geom type="mesh" mesh="vis_obj_{obj_str}" '
                    f'contype="0" conaffinity="0" group="2"/>'
                ]
                for i, pf in enumerate(part_files):
                    col_name = f'col_obj_{obj_str}_{i:02d}'
                    asset_lines.append(
                        f'    <mesh name="{col_name}" file="{pf}"/>'
                    )
                    geom_lines.append(
                        f'<geom type="mesh" mesh="{col_name}" '
                        f'contype="1" conaffinity="1" group="3" '
                        f'density="{OBJ_DENSITY}" '
                        f'friction="{OBJ_FRICTION}" '
                        f'solimp="{OBJ_SOLIMP}" solref="{OBJ_SOLREF}" '
                        f'condim="{OBJ_CONDIM}"/>'
                    )
                body_lines.append(f'''
              <body name="obj_{obj_str}" pos="{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}"
                    quat="{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}">
                <joint type="free" damping="1.0"/>
  {chr(10).join(geom_lines)}
              </body>''')
            else:
                if coacd and not part_files:
                    print(f'[warn] obj {obj_str}: no CoACD parts found, falling back to single mesh')
                asset_lines.append(
                    f'    <mesh name="mesh_obj_{obj_str}" file="{self.model_dir}/{obj_str}/nontextured.stl"/>'
                )
                body_lines.append(f'''
              <body name="obj_{obj_str}" pos="{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}"
                    quat="{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}">
                <joint type="free" damping="1.0"/>
                <geom type="mesh" mesh="mesh_obj_{obj_str}"
                      contype="1" conaffinity="1"
                      friction="{OBJ_FRICTION}"
                      solimp="{OBJ_SOLIMP}" solref="{OBJ_SOLREF}"
                      condim="{OBJ_CONDIM}" density="{OBJ_DENSITY}"/>
              </body>''')
        return asset_lines, body_lines

    def shape_xml(self, obj_indexes=None, coacd=False, strength='original'):
        asset_lines, body_lines = self._build_obj_lines(obj_indexes, coacd=coacd)

        xml = f"""<mujoco model="graspnet_scene_0000_mocap">
        <compiler angle="radian" meshdir="/" autolimits="true"/>

        <option integrator="implicitfast" impratio="10" cone="elliptic" timestep="0.002" noslip_iterations="3"/>

        <visual>
          <global offwidth="1280" offheight="720"/>
        </visual>

        <default>
          <default class="panda">
            <material specular="0.5" shininess="0.25"/>
            <joint armature="0.1" damping="1" axis="0 0 1" range="-2.8973 2.8973"/>
            <general dyntype="none" biastype="affine" ctrlrange="-2.8973 2.8973" forcerange="-87 87"/>
            <default class="finger">
              <joint axis="0 1 0" type="slide" range="0 0.04"/>
            </default>
            <default class="visual">
              <geom type="mesh" contype="0" conaffinity="0" group="2"/>
            </default>
            <default class="collision">
              <geom type="mesh" group="3" friction="5 1.0 0.1"/>
              <default class="fingertip_pad_collision_1">
                <geom type="box" size="0.0085 0.004 0.0085" pos="0 0.0055 0.0445"
                      friction="5 1.0 0.1" condim="6"
                      solimp="0.95 0.99 0.001" solref="0.005 1"/>
              </default>
              <default class="fingertip_pad_collision_2">
                <geom type="box" size="0.003 0.002 0.003" pos="0.0055 0.002 0.05"
                      friction="5 1.0 0.1" condim="6"
                      solimp="0.95 0.99 0.001" solref="0.005 1"/>
              </default>
              <default class="fingertip_pad_collision_3">
                <geom type="box" size="0.003 0.002 0.003" pos="-0.0055 0.002 0.05"
                      friction="5 1.0 0.1" condim="6"
                      solimp="0.95 0.99 0.001" solref="0.005 1"/>
              </default>
              <default class="fingertip_pad_collision_4">
                <geom type="box" size="0.003 0.002 0.0035" pos="0.0055 0.002 0.0395"
                      friction="5 1.0 0.1" condim="6"
                      solimp="0.95 0.99 0.001" solref="0.005 1"/>
              </default>
              <default class="fingertip_pad_collision_5">
                <geom type="box" size="0.003 0.002 0.0035" pos="-0.0055 0.002 0.0395"
                      friction="5 1.0 0.1" condim="6"
                      solimp="0.95 0.99 0.001" solref="0.005 1"/>
              </default>
            </default>
          </default>
        </default>

        <asset>
            <material class="panda" name="white"     rgba="1 1 1 1"/>
            <material class="panda" name="off_white" rgba="0.901961 0.921569 0.929412 1"/>
            <material class="panda" name="black"     rgba="0.25 0.25 0.25 1"/>

            <mesh name="hand_c"   file="{self.hand_assets}/hand.stl"/>
            <mesh name="hand_0"   file="{self.hand_assets}/hand_0.obj"/>
            <mesh name="hand_1"   file="{self.hand_assets}/hand_1.obj"/>
            <mesh name="hand_2"   file="{self.hand_assets}/hand_2.obj"/>
            <mesh name="hand_3"   file="{self.hand_assets}/hand_3.obj"/>
            <mesh name="hand_4"   file="{self.hand_assets}/hand_4.obj"/>
            <mesh name="finger_0" file="{self.hand_assets}/finger_0.obj"/>
            <mesh name="finger_1" file="{self.hand_assets}/finger_1.obj"/>

{chr(10).join(asset_lines)}
        </asset>

        <worldbody>
            <light directional="true" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>

            <!-- Table -->
            <body name="table" pos="0 0 -0.01">
              <geom type="box" size="0.5 0.5 0.01" rgba="0.8 0.7 0.6 1"
                    contype="1" conaffinity="1"/>
            </body>

            <body name="marker_tw" mocap="true" pos="0 0 0">
              <geom type="sphere" size="0.006" rgba="0.5 0.3 1 0.7"
                    contype="0" conaffinity="0" group="2"/>
            </body>
            <body name="marker_theo" mocap="true" pos="0 0 0">
              <geom type="sphere" size="0.006" rgba="0.85 0.35 0.2 0.7"
                    contype="0" conaffinity="0" group="2"/>
            </body>

            <!-- Mocap target: Python drives this via sim.mocap_pos / sim.mocap_quat -->
            <body name="hand_target" mocap="true" pos="0 0 0.6" quat="1 0 0 0">
              <geom type="sphere" size="0.006" rgba="1 0.3 0.3 0.4"
                    contype="0" conaffinity="0" group="2"/>
            </body>

            <!-- Hand: freejoint + weld equality to hand_target. Fingers remain dynamic. -->
            <body name="hand" childclass="panda" pos="0 0 0.6" quat="1 0 0 0">
                <freejoint name="hand_freejoint"/>
                <inertial mass="0.73" pos="-0.01 0 0.03" diaginertia="0.001 0.0025 0.0017"/>
                <geom mesh="hand_0" material="off_white" class="visual"/>
                <geom mesh="hand_1" material="black"     class="visual"/>
                <geom mesh="hand_2" material="black"     class="visual"/>
                <geom mesh="hand_3" material="white"     class="visual"/>
                <geom mesh="hand_4" material="off_white" class="visual"/>
                <geom mesh="hand_c" class="collision"/>
                <body name="left_finger" pos="0 0 0.0584">
                  <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                  <joint name="finger_joint1" class="finger"/>
                  <geom mesh="finger_0" material="off_white" class="visual"/>
                  <geom mesh="finger_1" material="black"     class="visual"/>
                  <geom mesh="finger_0" class="collision"/>
                  <geom class="fingertip_pad_collision_1"/>
                  <geom class="fingertip_pad_collision_2"/>
                  <geom class="fingertip_pad_collision_3"/>
                  <geom class="fingertip_pad_collision_4"/>
                  <geom class="fingertip_pad_collision_5"/>
                  <site name="left_tip"  pos="0 0.0015 0.0445" size="0.002" rgba="1 0 0 1"/>
                </body>
                <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                  <inertial mass="0.015" pos="0 0 0" diaginertia="2.375e-6 2.375e-6 7.5e-7"/>
                  <joint name="finger_joint2" class="finger"/>
                  <geom mesh="finger_0" material="off_white" class="visual"/>
                  <geom mesh="finger_1" material="black"     class="visual"/>
                  <geom mesh="finger_0" class="collision"/>
                  <geom class="fingertip_pad_collision_1"/>
                  <geom class="fingertip_pad_collision_2"/>
                  <geom class="fingertip_pad_collision_3"/>
                  <geom class="fingertip_pad_collision_4"/>
                  <geom class="fingertip_pad_collision_5"/>
                  <site name="right_tip" pos="0 0.0015 0.0445" size="0.002" rgba="0 1 0 1"/>
                </body>
            </body>

        <!-- Scene objects -->
        {''.join(body_lines)}
        </worldbody>

          <contact>
            <exclude body1="hand" body2="left_finger"/>
            <exclude body1="hand" body2="right_finger"/>
          </contact>

          <equality>
            <!-- Keep fingers symmetric -->
            <joint joint1="finger_joint1" joint2="finger_joint2"
                   solimp="0.95 0.99 0.001" solref="0.005 1"/>
            <!-- Weld hand body to mocap target -->
            <weld body1="hand_target" body2="hand"
                  solimp="0.95 0.99 0.001" solref="0.002 1"/>
          </equality>

          <tendon>
            <fixed name="split">
              <joint joint="finger_joint1" coef="0.5"/>
              <joint joint="finger_joint2" coef="0.5"/>
            </fixed>
          </tendon>

          <actuator>
{self._gripper_strength(strength)}
          </actuator>

        </mujoco>
        """
        return xml

    def save_xml(self, output_path, obj_indexes=None, coacd=False, strength='original'):
        with open(output_path, 'w') as f:
            f.write(self.shape_xml(obj_indexes, coacd=coacd, strength=strength))
        mode = 'with CoACD collision' if coacd else 'single-mesh collision'
        print(f'Saved → {output_path}  ({mode}, mocap-weld hand)')


def main():
    argparser = argparse.ArgumentParser(
        description='Generate MuJoCo XML (mocap-weld hand) for a GraspNet scene.')
    argparser.add_argument('--scene-dir', type=str, default=GRASPNET_SCENE_ROOT)
    argparser.add_argument('--model-dir', type=str, default=MODEL_DIR)
    argparser.add_argument('--hand-assets', type=str, default=HAND_ASSETS)
    argparser.add_argument('--camera', type=str, default=CAMERA)
    argparser.add_argument('--output-xml', type=str, default=OUTPUT_XML)
    argparser.add_argument('--obj-indexes', type=int, nargs='*', default=None,
                           help='Object indexes to include (default: all)')
    argparser.add_argument('--coacd', action='store_true',
                           help='Use CoACD convex parts for collision')
    argparser.add_argument('--strength', type=str, default='original',
                           choices=['original', 'strong'],
                           help='Gripper strength (default: original)')
    args = argparser.parse_args()

    scene = Scene(
        scene_dir=args.scene_dir,
        model_dir=args.model_dir,
        hand_assets=args.hand_assets,
        camera=args.camera,
    )
    scene.save_xml(args.output_xml, obj_indexes=args.obj_indexes, coacd=args.coacd, strength=args.strength)


if __name__ == '__main__':
    main()
