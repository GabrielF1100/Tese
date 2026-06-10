import os

# ===== CONFIGURATION =====
LABELS_FOLDER = "CercosporaLabels"

TARGET_CLASS = "2"  # Cercospora leaf spot

print("Filtering labels for class 2 (Cercospora leaf spot)...\n")

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

print("\n🎯 Only Cercospora leaf spot labels have been kept!")