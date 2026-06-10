import os
import cv2
import torch
from ultralytics import YOLO

# ==========================================================
# INITIAL CONFIGURATIONS
# ==========================================================
#
# In this stage, the following are defined:
# - the trained model for rust detection;
# - the folder containing the images to be analyzed;
# - the output folders where results will be stored.
#
# The input images correspond to the crops generated
# previously by the model responsible for identifying
# unhealthy regions of coffee leaves.
# ==========================================================

MODEL_PATH = "Rust_Model.pt"
INPUT_IMAGES_DIR = "Crops"
OUTPUT_IMAGES_DIR = "RustImages"
OUTPUT_LABELS_DIR = "RustLabels"

# Create output directories if they do not exist.
# This prevents errors during result saving.
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==========================================================
# MODEL LOADING
# ==========================================================
#
# The YOLO model previously trained to detect rust is
# loaded into memory to perform inference on images.
# ==========================================================

print("Loading model...")
model = YOLO(MODEL_PATH)

# ==========================================================
# PROCESSING DEVICE SELECTION
# ==========================================================
#
# If a CUDA-compatible GPU is available, it will be used
# to accelerate image processing.
# Otherwise, the CPU will be used.
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Using device: {device}")

# ==========================================================
# IMAGE LISTING
# ==========================================================
#
# Only valid image files will be processed.
# Other file types in the folder will be ignored.
# ==========================================================

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

files = [
    f for f in os.listdir(INPUT_IMAGES_DIR)
    if f.lower().endswith(valid_extensions)
]

print("\nStarting processing...\n")

# ==========================================================
# IMAGE PROCESSING
# ==========================================================
#
# Each image is sent to the YOLO model to detect
# the presence of rust symptoms.
#
# The results are used to:
# 1) Create images with bounding boxes;
# 2) Generate annotation files with detected coordinates.
# ==========================================================

for i, file_name in enumerate(files):

    image_path = os.path.join(INPUT_IMAGES_DIR, file_name)

    # Verify if file exists before processing
    if not os.path.isfile(image_path):
        continue

    print(f"[{i+1}/{len(files)}] Processing: {file_name}")

    # ======================================================
    # INFERENCE EXECUTION
    # ======================================================
    #
    # The conf=0.4 parameter ensures that only detections
    # with confidence ≥ 40% are considered.
    # ======================================================

    results = model(image_path, conf=0.4)
    r = results[0]

    # ======================================================
    # ANNOTATED IMAGE GENERATION
    # ======================================================
    #
    # The plot() method automatically draws:
    # - bounding boxes;
    # - class names;
    # - confidence scores.
    #
    # This allows quick visual evaluation of the model.
    # ======================================================

    annotated_image = r.plot()

    output_image_path = os.path.join(OUTPUT_IMAGES_DIR, file_name)

    cv2.imwrite(output_image_path, annotated_image)

    # ======================================================
    # LABEL FILE GENERATION
    # ======================================================
    #
    # For each image, a .txt file is created containing
    # the detected coordinates.
    #
    # Coordinates are stored in normalized YOLO format,
    # allowing reuse for training or further analysis.
    # ======================================================

    label_name = os.path.splitext(file_name)[0] + ".txt"
    label_path = os.path.join(OUTPUT_LABELS_DIR, label_name)

    # Original image dimensions provided by YOLO.
    h, w = r.orig_shape

    with open(label_path, "w") as f:

        # If detections exist, each one is saved
        # in the annotation file.
        if r.boxes is not None:

            for box in r.boxes:

                # Predicted class.
                cls = int(box.cls[0])

                # Detection confidence.
                conf = float(box.conf[0])

                # ==================================================
                # NORMALIZED COORDINATES
                # ==================================================
                #
                # xywhn returns:
                #
                # x_center -> center X coordinate
                # y_center -> center Y coordinate
                # width    -> box width
                # height   -> box height
                #
                # All values are normalized between 0 and 1
                # according to YOLO format.
                # ==================================================

                x_center, y_center, width, height = box.xywhn[0]

                # ==================================================
                # WRITE ANNOTATIONS
                # ==================================================
                #
                # Each line represents one detection:
                #
                # class x_center y_center width height confidence
                #
                # Example:
                #
                # 0 0.45 0.52 0.18 0.16 0.89
                #
                # ==================================================

                f.write(
                    f"{cls} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f} "
                    f"{conf:.6f}\n"
                )

# ==========================================================
# FINALIZATION
# ==========================================================
#
# At the end of processing, all analyzed images will have:
#
# - an annotated version with detections;
# - a corresponding label file.
#
# These results can be used for visual validation,
# metric computation, or dataset creation.
# ==========================================================

print("\n✅ Processing completed!")
print(f"Images: {OUTPUT_IMAGES_DIR}")
print(f"Labels: {OUTPUT_LABELS_DIR}")