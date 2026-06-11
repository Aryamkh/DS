import kagglehub
import shutil
import os

# Download
path = kagglehub.dataset_download("shivamb/netflix-shows")

# Target folder
target_dir = "data"

# Create folder if not exists
os.makedirs(target_dir, exist_ok=True)

# Move downloaded files into ./data
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), target_dir)

print("Saved to:", target_dir)