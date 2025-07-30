#!/usr/bin/python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from point_cloud_viz import process_pointcloud

# === Paths ===
bin_path = "/home/havee005/Velo_Vision/src/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/velodyne_points/data/0000000182.bin"
img_path = "/home/havee005/Velo_Vision/src/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000182.png"

# === Load data ===
_, _, dbscan_labels, non_ground_points = process_pointcloud(bin_path)
img = cv2.imread(img_path)

# === Calibration Matrices ===
P_rect = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
                   [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

R_rect = np.array([[0.99992475, 0.00975976, -0.00734152],
                   [-0.0097913,  0.99994262, -0.00430371],
                   [0.00729911, 0.0043753,   0.99996319]])

T_velo2cam = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                       [1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],
                       [9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],
                       [0.0,           0.0,            0.0,          1.0]])

# === 4x4 Rectification ===
R_rect_4x4 = np.eye(4)
R_rect_4x4[:3, :3] = R_rect 

# === Transform to Camera Frame ===
points_hom = np.hstack((non_ground_points, np.ones((non_ground_points.shape[0], 1))))  # Nx4
cam_points = (R_rect_4x4 @ T_velo2cam @ points_hom.T)[:3, :]  # 3xN

# === Filter in-front points ===
in_front = cam_points[2, :] > 0
cam_points = cam_points[:, in_front]
labels = dbscan_labels[in_front]

# === Projection ===
proj_points = P_rect @ np.vstack((cam_points, np.ones((1, cam_points.shape[1]))))  # 3xN
proj_points /= proj_points[2, :]

u = proj_points[0, :].astype(int)
v = proj_points[1, :].astype(int)
u-=5

# === Color mapping ===
cmap = plt.get_cmap("tab20")
max_label = labels.max() if labels.max() > 0 else 1
colors = cmap(labels / max_label)[:, :3] * 255
colors = colors.astype(np.uint8)

# === Draw on image ===
for x, y, color, label in zip(u, v, colors, labels):
    if label == -1:
        continue
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (x, y), 2, tuple(int(c) for c in color), -1)

# === Show image
cv2.imshow("Clustered LiDAR Projection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
