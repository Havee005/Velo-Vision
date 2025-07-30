#!/usr/bin/python3

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def run_ransac_segmentation(pcd, distance_threshold=0.1, ransac_n=3, iterations=100):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=iterations
    )
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    return ground, non_ground

def rotation_matrix(deg_x, deg_y, deg_z):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(deg_x)), -np.sin(np.deg2rad(deg_x))],
        [0, np.sin(np.deg2rad(deg_x)),  np.cos(np.deg2rad(deg_x))]
    ])
    Ry = np.array([
        [np.cos(np.deg2rad(deg_y)), 0, np.sin(np.deg2rad(deg_y))],
        [0, 1, 0],
        [-np.sin(np.deg2rad(deg_y)), 0, np.cos(np.deg2rad(deg_y))]
    ])
    Rz = np.array([
        [np.cos(np.deg2rad(deg_z)), -np.sin(np.deg2rad(deg_z)), 0],
        [np.sin(np.deg2rad(deg_z)),  np.cos(np.deg2rad(deg_z)), 0],
        [0, 0, 1]
    ])
    return Rz @ Rx @ Ry

def process_pointcloud(bin_file_path):
    # Load points
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Voxel-Grid Downsampling
    # Didn't use coz it's downsampling points many points even when the
    # thershold is too low, eventually affecting RANSAC to segment the plane(Ground_points).

    # RANSAC for ground plane
    ground, non_ground = run_ransac_segmentation(pcd)
    ground.paint_uniform_color([0.01, 0.01, 0.01])
    non_ground.paint_uniform_color([0.3, 1.0, 0.6])

    # DBSCAN Clustering
    dbscan_labels = np.array(non_ground.cluster_dbscan(
        eps=0.6, min_points=8, print_progress=False))
    
    xyz_nonground = np.asarray(non_ground.points)

    return ground, non_ground, dbscan_labels, xyz_nonground


def draw_car_bounding_boxes(non_ground, dbscan_labels, vis):
    max_label = dbscan_labels.max()
    cmap = plt.get_cmap("tab20")
    cluster_colors = cmap(dbscan_labels / (max_label if max_label > 0 else 1))
    cluster_colors[dbscan_labels < 0] = 0
    non_ground.colors = o3d.utility.Vector3dVector(cluster_colors[:, :3])

    for i in range(max_label + 1):
        indices = np.where(dbscan_labels == i)[0]
        if len(indices) < 10:
            continue
        cluster = non_ground.select_by_index(indices)
        obb = cluster.get_axis_aligned_bounding_box()
        dims = sorted(obb.get_extent(), reverse=True)
        length, width, height = dims
        if 3.0 < length < 7.0 and 1.2 < width < 4.5 and 1.2 < height < 3.5:
            obb.color = (1, 0, 0)
            vis.add_geometry(obb)


# === Main script ===
if __name__ == "__main__":
    bin_path = "/home/havee005/Velo_Vision/src/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/velodyne_points/data/0000000321.bin"

    # Step 1: Process
    ground, non_ground, dbscan_labels, _ = process_pointcloud(bin_path)

    # Step 2: Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud")

    vis.add_geometry(ground)
    vis.add_geometry(non_ground)
    draw_car_bounding_boxes(non_ground, dbscan_labels, vis)

    # Step 3: Set camera
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    R = rotation_matrix(180, 90, -90)
    T = np.array([[-1], [-0.25], [1]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = T
    cam_params.extrinsic = extrinsic
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

    # Step 4: Run viewer
    vis.run()
    vis.destroy_window()
