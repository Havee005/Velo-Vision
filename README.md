# Velo-Vision
Velo-Vision: Bridging 3D vision and 2D perception for autonomous mobility.

This project focuses on multi-sensor fusion, specifically projecting 3D LiDAR point clouds onto 2D camera images using data from the KITTI Vision Benchmark Suite, with the goal of identifying and localizing vehicles in urban scenes through geometric processing and unsupervised clustering. It delivers a fully functional pipeline that visualizes these projected LiDAR clusters over RGB frames, enabling intuitive perception of 3D environments from a 2D perspective — a core capability for autonomous vehicles and robotic navigation.


## 1. Concept

Late fusion is a sensor fusion technique where information from different modalities—like LiDAR and camera images—is first processed independently to extract high-level features (e.g., object clusters or bounding boxes). These features are then combined at a later stage to make final decisions, such as object detection or scene understanding.

## Camera
### 1.1. Vehicle Detection and Tracking

![Vehicle_detection gif](https://github.com/user-attachments/assets/1a252de4-2dcb-4182-bb25-f906cbc81e39)

- Utilized a pretrained YOLOv4 model to detect vehicles in 2D camera frames from the KITTI dataset, leveraging COCO-trained weights for fast and accurate results without retraining.
- Processed KITTI's left RGB camera's image sequences to for detection and tracking evaluation.
- Tracking the detected vehicle by creating the bounding boxes around it.

## 3D LiDAR

### - 3D Point Cloud
![3D_pointcloud](https://github.com/user-attachments/assets/a815dcbc-0f1c-46c8-bda3-86173cb80a73)

### 1.2 3D Point Segmentation 
![Segmentation](https://github.com/user-attachments/assets/27d66ed3-1434-486f-b2e0-d5707c10f68f)

- Applied RANSAC (Random Sample Consensus) to segment the dominant ground plane from raw 3D LiDAR point clouds, effectively isolating road surfaces in urban scenes.
- By removing the ground plane, non-ground points—primarily vehicles, poles and other obstacles—were separated for further analysis and clustering.
- RANSAC was chosen due to its robustness against noise and sparse outliers, making it ideal for unstructured outdoor LiDAR data.
- This segmentation step lays the foundation for downstream tasks like clustering and 3D bounding box fitting, ensuring only relevant obstacles are processed.

### 1.3 3D Point Clustering
![Clustering](https://github.com/user-attachments/assets/9ee42237-ec4b-4449-b056-74b2e301ef63)

- After ground removal, DBSCAN identifies dense point regions—likely vehicles or obstacles—by exploiting spatial proximity. Enabling accurate localization of individual objects.
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups the segmented non-ground LiDAR points into  object clusters without requiring labeled data.

