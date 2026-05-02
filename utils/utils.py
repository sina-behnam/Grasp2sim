import numpy as np
import open3d as o3d
from IPython.display import Image

def save_point_cloud_to_ply(geometries, filename):

    try:
        combined = o3d.geometry.PointCloud()
    
        for geom in geometries:
            if isinstance(geom, o3d.geometry.PointCloud):
                combined += geom
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                combined += geom.sample_points_uniformly(number_of_points=1000)
    
        o3d.io.write_point_cloud(filename, combined)

    except Exception as e:
        print(f"Error saving point cloud: {e}")
        return False
    
    return True

def render_graspnet_scene_notebook(geometries, width=1280, height=720, shift_x=-0.2, save_path="scene.png"):
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

def render_geometries_to_notebook(
    geometries,
    width=1280, height=720,
    view='top',           # 'top', 'front', 'side', 'iso', or custom
    zoom=1.2,             # >1 zooms out, <1 zooms in
    shift=(0, 0, 0),      # nudge the look-at point
    bg=(1, 1, 1, 1),
    save_path="scene.png",
    show_axes=True,
):

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(list(bg))

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    geoms_to_render = list(geometries)
    if show_axes:
        # 10 cm axes at world origin (red=X, green=Y, blue=Z)
        geoms_to_render.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        )

    for i, g in enumerate(geoms_to_render):
        renderer.scene.add_geometry(f"geom_{i}", g, mat)

    # bounds from the actual scene contents (not axes)
    bounds = o3d.geometry.AxisAlignedBoundingBox()
    for g in geometries:
        bounds += g.get_axis_aligned_bounding_box()
    center = bounds.get_center() + np.array(shift)
    diag = np.linalg.norm(bounds.get_extent())
    dist = diag * zoom

    # Common camera presets (table-frame conventions: Z up)
    presets = {
        'top':   (np.array([0, 0,  dist]), np.array([0, 1, 0])),
        'front': (np.array([0, -dist, 0]), np.array([0, 0, 1])),
        'side':  (np.array([dist, 0, 0]),  np.array([0, 0, 1])),
        'iso':   (np.array([dist, -dist, dist]) * 0.6, np.array([0, 0, 1])),
    }
    eye_offset, up = presets.get(view, presets['iso'])
    eye = center + eye_offset

    renderer.setup_camera(
        60.0,
        center.astype(np.float32),
        eye.astype(np.float32),
        up.astype(np.float32),
    )

    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    return Image(save_path)
