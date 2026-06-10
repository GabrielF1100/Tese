"""
Initially, a pre-trained YOLO model for identifying unhealthy leaves was used to detect regions of interest in the training dataset images. For each detection
belonging to the "Unhealthy" class, the region defined by the bounding box was cropped. The generated crops were stored in a new folder and accompanied by YOLO-format
annotation files. Since each crop contains only the detected region, the annotations were defined so that the object fully occupies the image (center at 0.5; 0.5
and width and height equal to 1.0). This procedure enabled the creation of a new dataset containing exclusively samples of unhealthy regions, facilitating later
stages of disease training and classification in coffee plants.

Automatic script for extracting regions classified as
"Unhealthy" using a trained YOLO model.

Objective:
- Detect unhealthy regions in images.
- Crop each detected region.
- Save the crops in a new folder.
- Generate YOLO-format annotation files for each crop.

Author: Gabriel Felippe Baptista
"""

import os
import cv2
from ultralytics import YOLO

# ==========================================================
# CONFIGURATIONS
# ==========================================================

# Folder containing the original images
INPUT_DIR = "Dataset"

# Folder where crops will be stored
OUTPUT_CROP_DIR = "Crops"

# Path to the trained model
MODEL_PATH = "Unhealthy_Model"

# Class ID corresponding to "Unhealthy"
NON_HEALTHY_CLASS_ID = 1

# Minimum confidence threshold for accepting a detection
CONF_THRESHOLD = 0.25

# Create output folder if it does not exist
os.makedirs(OUTPUT_CROP_DIR, exist_ok=True)

# ==========================================================
# MODEL LOADING
# ==========================================================

print("Loading model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!\n")

# ==========================================================
# IMAGE PROCESSING
# ==========================================================

for image_name in os.listdir(INPUT_DIR):

    # Ignore non-image files
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_DIR, image_name)

    # Read image
    image = cv2.imread(image_path)

    # If image cannot be loaded
    if image is None:
        print(f"[ERROR] Could not open: {image_name}")
        continue

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Run YOLO inference
    results = model(image)

    # Counter for naming crops
    crop_count = 0

    # ======================================================
    # RESULT ANALYSIS
    # ======================================================

    for result in results:

        # If no detections exist
        if result.boxes is None:
            continue

        # Iterate through all detected boxes
        for box in result.boxes:

            # Detected class
            class_id = int(box.cls[0])

            # Detection confidence
            confidence = float(box.conf[0])

            # ==================================================
            # FILTER ONLY "UNHEALTHY" CLASS
            # ==================================================

            if (
                class_id == NON_HEALTHY_CLASS_ID
                and confidence >= CONF_THRESHOLD
            ):

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(image_width, x2)
                y2 = min(image_height, y2)

                # ==================================================
                # CROP DETECTED REGION
                # ==================================================

                crop = image[y1:y2, x1:x2]

                # Ignore invalid crops
                if crop.size == 0:
                    continue

                # ==================================================
                # SAVE CROP IMAGE
                # ==================================================

                crop_filename = (
                    f"{os.path.splitext(image_name)[0]}"
                    f"_unhealthy_{crop_count}.jpg"
                )

                crop_path = os.path.join(
                    OUTPUT_CROP_DIR,
                    crop_filename
                )

                cv2.imwrite(crop_path, crop)

                # ==================================================
                # GENERATE YOLO ANNOTATION FILE
                # ==================================================
                #
                # Since the object occupies the entire crop,
                # the annotation is defined as:
                #
                # class = 1
                # center_x = 0.5
                # center_y = 0.5
                # width = 1.0
                # height = 1.0
                #
                # This indicates the object fills the entire image.
                #
                # ==================================================

                label_filename = (
                    f"{os.path.splitext(crop_filename)[0]}.txt"
                )

                label_path = os.path.join(
                    OUTPUT_CROP_DIR,
                    label_filename
                )

                with open(label_path, "w") as label_file:
                    label_file.write(
                        f"{NON_HEALTHY_CLASS_ID} 0.5 0.5 1.0 1.0\n"
                    )

                crop_count += 1

    print(
        f"[OK] {image_name} -> "
        f"{crop_count} unhealthy regions extracted."
    )

print("\nProcessing completed!")