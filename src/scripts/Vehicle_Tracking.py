#!/usr/bin/python3

import cv2
import os
from Vehicle_Detection import ObjectDetection
# import time 

# t_prev = time.time()

# Path to KITTI left camera images
kitti_image_dir = "/home/havee005/Velo_Vision/src/2011_09_29_drive_0004_sync/2011_09_29/2011_09_29_drive_0004_sync/image_02/data"
image_files = sorted([f for f in os.listdir(kitti_image_dir) if f.endswith(".png")])

ob = ObjectDetection()

count = 0

for image_file in image_files:
    image_path = os.path.join(kitti_image_dir, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Warning: Could not read {image_path}")
        continue

    (class_ids, scores, boxes) = ob.detect(frame=frame)
    for box in boxes:
        (x, y, w, h) = box
        count += 1
        cv2.rectangle(frame, (x, y), (w+x, h+y), (30, 255, 156), 2)

    cv2.imshow("Vehicle Detection", frame)

    key = cv2.waitKey(50)  # Wait 0.5 seconds
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()
# print((time.time() - t_prev) / 60)
