import os
import cv2
import torch
from ultralytics import YOLO

# ==========================================================
# INITIAL CONFIGURATIONS
# ==========================================================
#
# In this stage, the following are defined:
# - the trained model for mite detection;
# - the folder containing the images to be processed;
# - the output folders where results will be stored.
#
# The input images correspond to the crops generated
# previously by the model responsible for identifying
# unhealthy regions of coffee leaves.
# ==========================================================

MODEL_PATH = "Mite_Model.pt"
INPUT_IMAGES_DIR = "Crops"
OUTPUT_IMAGES_DIR = "MiteImages"
OUTPUT_LABELS_DIR = "MiteLabels"

# Create output directories if they do not exist.
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# ==========================================================
# MODEL LOADING
# ==========================================================

print("Loading model...")
model = YOLO(MODEL_PATH)

# ==========================================================
# PROCESSING DEVICE SELECTION
# ==========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"Using device: {device}")

# ==========================================================
# IMAGE LISTING
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

for i, file_name in enumerate(files):

    image_path = os.path.join(INPUT_IMAGES_DIR, file_name)

    if not os.path.isfile(image_path):
        continue

    print(f"[{i+1}/{len(files)}] Processing: {file_name}")

    # ======================================================
    # INFERENCE EXECUTION
    # ======================================================

    results = model(image_path, conf=0.4)
    r = results[0]

    # ======================================================
    # ANNOTATED IMAGE GENERATION
    # ======================================================

    annotated_image = r.plot()

    output_image_path = os.path.join(OUTPUT_IMAGES_DIR, file_name)

    cv2.imwrite(output_image_path, annotated_image)

    # ======================================================
    # LABEL FILE GENERATION
    # ======================================================

    label_name = os.path.splitext(file_name)[0] + ".txt"
    label_path = os.path.join(OUTPUT_LABELS_DIR, label_name)

    with open(label_path, "w") as f:

        if r.boxes is not None:

            for box in r.boxes:

                # Predicted class from the model.
                cls = int(box.cls[0])

                # Detection confidence.
                conf = float(box.conf[0])

                # YOLO normalized coordinates.
                x_center, y_center, width, height = box.xywhn[0]

                # Write in YOLO format with confidence.
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

print("\n✅ Processing completed!")
print(f"Images: {OUTPUT_IMAGES_DIR}")
print(f"Labels: {OUTPUT_LABELS_DIR}")