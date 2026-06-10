import os

# This script is used in Stage III with the purpose of merging all labels
# that were separated in Stage II.

# ===== BASE PATH =====
BASE = "Crops"

CERCOSPORA_FOLDER = os.path.join(BASE, "CercosporaLabels")
RUST_FOLDER = os.path.join(BASE, "RustLabels")
MITE_FOLDER = os.path.join(BASE, "MiteLabels")

OUTPUT_FOLDER = os.path.join(BASE, "Labels")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Merging labels...\n")

# ===== PROCESSING ORDER =====
folders = [
    CERCOSPORA_FOLDER,
    RUST_FOLDER,
    MITE_FOLDER
]

# collect all file names
files = set()

for folder in folders:
    files.update(
        f for f in os.listdir(folder)
        if f.endswith(".txt")
    )

# merge contents
for file_name in files:

    all_lines = []

    for folder in folders:
        file_path = os.path.join(folder, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_lines.extend(f.readlines())

    output