#!/usr/bin/python3

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
# import time
from ProcessPointCloud import PointCloudProcessing as PCP

# t_prev = time.time()

# === KITTI point cloud directory ===
kitti_pcd_dir = "/home/havee005/Velo_Vision/src/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/velodyne_points/data"
bin_files = sorted([f for f in os.listdir(kitti_pcd_dir) if f.endswith(".bin")])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="KITTI Viewer")

# Pre-load first frame to initialize pcd
first_bin = os.path.join(kitti_pcd_dir, bin_files[0])
points = np.fromfile(first_bin, dtype=np.float32).reshape(-1, 4)
xyz = points[:, :3]
intensity = points[:, 3]
intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
cmap = plt.get_cmap('jet')
colors = cmap(intensity_norm)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd)


# === Setup camera view ===
view_ctl = vis.get_view_control()
cam_params = view_ctl.convert_to_pinhole_camera_parameters()
R = PCP.rotation_matrix(180, 90, -90)
T = np.array([[0], [0.25], [1]])  # Camera at x=-1, y=-0.25, z=1
extrinsic = np.eye(4)
extrinsic[:3, :3] = R
extrinsic[:3, 3:] = T
cam_params.extrinsic = extrinsic
view_ctl.convert_from_pinhole_camera_parameters(cam_params)

# === Loop through frames ===
for bin_file in bin_files[1:]:
    pcd_path = os.path.join(kitti_pcd_dir, bin_file)
    points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    intensity = points[:, 3]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    colors = cmap(intensity_norm)[:, :3]

    pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 1. Voxel-Grid Downsampling
    # Didn't use coz it's downsampling points many points even when the
    # thershold is too low, eventually affecting RANSAC to segment the plane(Ground_points).

    # 2. RANSAC Segmentation
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.1,
                                            ransac_n=3,
                                            num_iterations=100) # Open3D RANSAC
    # plane_model, inliers = PCP.ransac_plane(xyz, threshold=0.1, iterations=250) # Own RANSAC

    ground = pcd.select_by_index(inliers)   # Ground points
    ground.paint_uniform_color([0.01, 0.01, 0.01])

    non_ground = pcd.select_by_index(inliers, invert=True) # Non-ground
    non_ground.paint_uniform_color([0.3, 1.0, 0.6]) 

    # 3. DBSCAN Clustering
    dbscan_labels = np.array(non_ground.cluster_dbscan(eps=0.7, min_points=15,print_progress=False))
    # print(f"Found {dbscan_labels.max() + 1} object clusters (label ≥ 0)")

    # 4. Combine ground and non-ground point clouds
    full_scene = ground + non_ground
    pcd.points = full_scene.points
    pcd.colors = full_scene.colors

vis.destroy_window()
# print((time.time() - t_prev) / 60)
