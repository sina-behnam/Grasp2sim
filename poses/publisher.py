import numpy as np
import os
from tqdm import tqdm

from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI import GraspGroup
import open3d as o3d
import copy

TOTAL_SCENE_NUM = 190
GRASP_HEIGHT = 0.02
FRICTION_COEF_THRESH = 0.4

CAMERA_POSES = '/home/sbehnam/Project/data/scenes/scene_0000/kinect/camera_poses.npy'
CAMERA_2_TABLE = '/home/sbehnam/Project/data/scenes/scene_0000/kinect/cam0_wrt_table.npy'

COLLISION_LABEL_PATH = '/home/sbehnam/Project/data/scenes/scene_0000/collision_labels.npz'
GRASP_LABEL_PATH     = '/home/sbehnam/Project/data/scenes/scene_0000/grasp_labels/'
FIRST_ANNOTATION_XML = '/home/sbehnam/Project/data/scenes/scene_0000/kinect/annotations/0000.xml'

def loadCollisionLabel(collision_label_path):
    print("Loading collision labels ...")
    collision_label = np.load(collision_label_path)
    collisionLabel = []
    for j in range(len(collision_label)):
        collisionLabel.append(collision_label['arr_{}'.format(j)])
    return collisionLabel

def load_grasp_label(grasp_label_path, object_ids):
    graspLabels = {}    
    for i in tqdm(object_ids, desc="Loading grasp labels ..."):
        file = np.load(
            os.path.join(
                grasp_label_path,
                '{}_labels.npz'.format(str(i).zfill(3))
            )
        )
        graspLabels[i] = (
            file['points'].astype(np.float32),
            file['offsets'].astype(np.float32),
            file['scores'].astype(np.float32)
        )
    return graspLabels

def cam_pose(camera_poses_path, camera_2_table_path):
    camera_poses = np.load(camera_poses_path)
    align_mat    = np.load(camera_2_table_path)
    return align_mat @ camera_poses[0]  # Always use the first camera pose !!!

def pose2vectors(annotation_xml_path):
    xml_reader  = xmlReader(annotation_xml_path)
    return xml_reader.getposevectorlist()

def publish_grasp_labels(wanted_object_ids : list):
    camera_pose = np.load(CAMERA_POSES)[0]
    pose_vectors = pose2vectors(FIRST_ANNOTATION_XML)

    obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)

    grasp_labels = load_grasp_label(GRASP_LABEL_PATH, obj_list)
    collision_dump = loadCollisionLabel(COLLISION_LABEL_PATH)

    num_views, num_angles, num_depths = 300, 12, 4
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

    grasp_group = GraspGroup()
    for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
        if obj_idx not in wanted_object_ids:
            continue
        sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
        collision = collision_dump[i]
        point_inds = np.arange(sampled_points.shape[0])

        num_points = len(point_inds)
        target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
        target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
        views = np.tile(template_views, [num_points, 1, 1, 1, 1])
        angles = offsets[:, :, :, :, 0]
        depths = offsets[:, :, :, :, 1]
        widths = offsets[:, :, :, :, 2]

        mask1 = ((fric_coefs <= FRICTION_COEF_THRESH) & (fric_coefs > 0) & ~collision)
        target_points = target_points[mask1]
        target_points = transform_points(target_points, trans)
        target_points = transform_points(target_points, np.linalg.inv(camera_pose))
        views = views[mask1]
        angles = angles[mask1]
        depths = depths[mask1]
        widths = widths[mask1]
        fric_coefs = fric_coefs[mask1]

        Rs = batch_viewpoint_params_to_matrix(-views, angles)
        Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)
        Rs = np.matmul(np.linalg.inv(camera_pose)[np.newaxis,:3,:3], Rs)

        num_grasp = widths.shape[0]
        scores = (1.1 - fric_coefs).reshape(-1,1)
        widths = widths.reshape(-1,1)
        heights = GRASP_HEIGHT * np.ones((num_grasp,1))
        depths = depths.reshape(-1,1)
        rotations = Rs.reshape((-1,9))
        object_ids = obj_idx * np.ones((num_grasp,1), dtype=np.int32)

        obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(np.float32)

        grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))

    return grasp_group

def render_geometries_to_notebook(geometries, width=1280, height=720, shift_x=-0.2, save_path="scene.png"):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    
    for i, g in enumerate(geometries):
        renderer.scene.add_geometry(f"geom_{i}", g, mat)
    
    bounds = o3d.geometry.AxisAlignedBoundingBox()
    for g in geometries:
        bounds += g.get_axis_aligned_bounding_box()
    
    center = bounds.get_center()
    extent = bounds.get_extent()
    look_at = center + np.array([extent[0] * shift_x, 0, 0])
    eye = look_at + np.array([0, 0, -np.linalg.norm(extent) * 0.35])
    up = np.array([0, 1, 0])
    
    renderer.setup_camera(60.0, look_at.astype(np.float32),
                          eye.astype(np.float32), up.astype(np.float32))
    
    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    
    from IPython.display import Image
    return Image(save_path)

def gg_filter_by_object_id(grasps_group : GraspGroup, object_id) -> GraspGroup:

    filtered_grasp_group_array = copy.deepcopy(grasps_group.grasp_group_array)

    filter_grasp_group = GraspGroup(filtered_grasp_group_array[filtered_grasp_group_array[:, 16] == object_id])

    return filter_grasp_group

    