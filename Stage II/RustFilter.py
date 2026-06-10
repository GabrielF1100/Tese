import os

# ===== CONFIGURATION =====
LABELS_FOLDER = "RustLabels"

print("Filtering labels for class 0 (rust)...\n")

files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith(".txt")]

for file_name in files:

    file_path = os.path.join(LABELS_FOLDER, file_name)

    with open(file_path, "r") as f:
        lines = f.readlines()

    # keep only class 0
    filtered_lines = [
        line for line in lines
        if line.strip().split()[0] == "0"
    ]

    # overwrite file
    with open(file_path, "w") as f:
        f.writelines(filtered_lines)

    print(f"✅ Processed: {file_name}")

print("\n🎯 Only rust labels have been kept!")