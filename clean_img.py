import os
from PIL import Image

dataset_path = "dataset/asl_alphabet_train"  # Adjust if your folder is named differently

for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except:
                print(f"❌ Deleting corrupted or non-image file: {file_path}")
                os.remove(file_path)