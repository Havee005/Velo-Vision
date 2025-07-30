# Velo-Vision
Velo-Vision: Bridging 3D vision and 2D perception for autonomous mobility.

This project focuses on multi-sensor fusion, specifically projecting 3D LiDAR point clouds onto 2D camera images using data from the KITTI Vision Benchmark Suite, with the goal of identifying and localizing vehicles in urban scenes through geometric processing and unsupervised clustering. It delivers a fully functional pipeline that visualizes these projected LiDAR clusters over RGB frames, enabling intuitive perception of 3D environments from a 2D perspective — a core capability for autonomous vehicles and robotic navigation.


# Concept

Late fusion is a sensor fusion technique where information from different modalities—like LiDAR and camera images—is first processed independently to extract high-level features (e.g., object clusters or bounding boxes). These features are then combined at a later stage to make final decisions, such as object detection or scene understanding.

# Working
## 1. Camera
### 1.1. Vehicle Detection and Tracking
![Vehicle_detection gif](https://github.com/user-attachments/assets/1a252de4-2dcb-4182-bb25-f906cbc81e39)

- Utilized a pretrained YOLOv4 model to detect vehicles in 2D camera frames from the KITTI dataset, leveraging COCO-trained weights for fast and accurate results without retraining.
- Processed KITTI's left RGB camera's image sequences to for detection and tracking evaluation.
- Tracking the detected vehicle by creating the bounding boxes around it.

## 2. 3D LiDAR

### 2.1. 3D Point Cloud
<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/b088fccc-14a8-41c8-8e6f-c3bb0c95d359" alt="PCD" width="475"/>
  <img src="https://github.com/user-attachments/assets/a815dcbc-0f1c-46c8-bda3-86173cb80a73" alt="3D_pointcloud" width="500"/>
</div>

### 2.2. 3D Point Segmentation 
![Segmentation](https://github.com/user-attachments/assets/27d66ed3-1434-486f-b2e0-d5707c10f68f)

- Applied RANSAC (Random Sample Consensus) to segment the dominant ground plane from raw 3D LiDAR point clouds, effectively isolating road surfaces in urban scenes.
- By removing the ground plane, non-ground points—primarily vehicles, poles and other obstacles—were separated for further analysis and clustering.
- RANSAC was chosen due to its robustness against noise and sparse outliers, making it ideal for unstructured outdoor LiDAR data.
- This segmentation step lays the foundation for downstream tasks like clustering and 3D bounding box fitting, ensuring only relevant obstacles are processed.

### 2.3. 3D Point Clustering
![Clustering](https://github.com/user-attachments/assets/9ee42237-ec4b-4449-b056-74b2e301ef63)

- After ground removal, DBSCAN identifies dense point regions, likely vehicles or obstacles, by exploiting spatial proximity. Enabling accurate localization of individual objects.
- DBSCAN (Density Based Spatial Clustering of Applications with Noise) groups the segmented non-ground LiDAR points into object clusters without requiring labeled data.

### 2.4. Bounding Box
<img width="1825" height="1017" alt="Bounding_box" src="https://github.com/user-attachments/assets/7291a75f-a0d3-46da-8de3-30c7d7d94da3" />

- After clustering, bounding boxes are fitted to each detected object. The project uses axis-aligned bounding boxes (AABB) to wrap object clusters efficiently.
- These fitted boxes can be extended to track objects across frames or matched with image based detections for late sensor fusion.

## 3. Late Fusion

### 3.1. Sensor Calibration
- Intrinsic parameters: Defines the internal characteristics of the camera such as focal length, principal point and skew. They are used by the camera projection matrix which is essential for projecting 3D points onto a 2D image plane.
- Rectification matrix: A rotation matrix aligning the camera coordinate system to the ideal stereo image plane so that epipolar lines in both stereo images are parallel.
- The extrinsic calibration of a LiDAR sensor and camera estimates a rigid transformation between them that establishes a geometric relationship between their coordinate systems.

<p align="center">
  <img width="360" height="240" alt="Projection_Matrix" src="https://github.com/user-attachments/assets/7993c79d-ee43-4f58-8699-548eb87c7b71" />
</p>

### 3.2. 3D to 2D Projection (LiDAR2Cam)
- The 3D LiDAR points are first transformed from the LiDAR coordinate frame to the camera coordinate frame using the extrinsic parameters.
- The transformed 3D points are then projected onto the 2D image plane using the camera’s projection matrix. Resulting in corresponding pixel coordinates for overlaying on the image.

<p align="center">
<img width="1238" height="374" alt="Clustered LiDAR2CAM Projection " src="https://github.com/user-attachments/assets/b5f89136-34de-4c37-a047-326b86526b8c" />
</p>

# Code Explanation
- Used Libraries:
  - [CV2 (Open CV)](https://pypi.org/project/opencv-python/)
  - [Open3D](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://pypi.org/project/open3d/&ved=2ahUKEwi_nvrB3-OOAxXlAHkGHUunHP0QFnoECCAQAQ&usg=AOvVaw3CP-xeew6yx8Cu3fWGEf1n)
  - YOLO Object detection model
  - KITTI: LiDAR & CAMERA sync data
  - Matplotlib
  - numpy
 
### Vehicle_Detection
- Using pre-trained object detection model which has 80 different classes.
- Creates bounding box around detected car.
### Vehilcle_Tracking
- Detects the car and other vehicles when the KITTI image frame is passed using openCV sequencially (326-frames -- left camera's image). 
- Tracks the vehicle continuously by creating bounding box around the vehicle by using Vehicle_Detection.py file which is a module.
### Point_cloud_viz
- This is a python module used to visualize the KITTI's LiDAR data using Open3D library.
- This file process the PCD data by RANSAC Segmenting -> DBSCAN Clustering -> Bounding box.
### Point_cloud_viz2
- This file does the same -- processes the PCD but by looping all .bin (PCD format) files sequentially.
### velo2cam_projection
- Gets the both image and pcd data as input.
- Processes the PCD data using point_cloud_viz.py module which outputs the clustered XYZ coordinates of the points.
- Using Transformation, that is by using calibrationand projection matrix of LiDAR to camera frame the clustered points of LiDAR are projected onto the 2D image frame.
- As a result it outputs the clustered point cloud data onto the 2D frame on which vehicles were detected.
