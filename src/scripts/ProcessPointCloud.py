import random
import numpy as np
import open3d as o3d

class PointCloudProcessing:

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
    

    def ransac_plane(xyz, threshold, iterations):
        best_inliers = []
        best_eq = None
        n_points = xyz.shape[0]

        for _ in range(iterations):
            # Randomly sample 3 points
            samples = xyz[np.random.choice(n_points, 3, replace=False)]
            p1, p2, p3 = samples

            # Compute plane normal vector
            normal = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(normal) == 0:
                continue  # Degenerate case, skip

            normal = normal / np.linalg.norm(normal)
            a, b, c = normal
            d = -np.dot(normal, p1)

            # Compute distances of all points to the plane
            distances = np.abs((xyz @ normal) + d)

            # Find inliers
            inliers = np.where(distances < threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_eq = (a, b, c, d)

        return best_eq, best_inliers
