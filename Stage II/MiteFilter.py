import os

# ===== CONFIGURATION =====
LABELS_FOLDER = "MiteLabels"

TARGET_CLASS = "1"  # Mite

print("Filtering labels for class 1 (Mite)...\n")

for file_name in os.listdir(LABELS_FOLDER):

    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(LABELS_FOLDER, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    filtered_lines = [
        line for line in lines
        if line.strip().split()[0] == TARGET_CLASS
    ]

    with open(file_path, "w") as f:
        f.writelines(filtered_lines)

    print(f"✅ {file_name}")

print("\n🎯 Only Mite labels have been kept!")