import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("kaustubhdhote/human-faces-dataset")

print("Path to dataset files:", path)
real_images="Real Images"

real_dir=os.path.join(path,"Human Faces Dataset",real_images)

jpg_files = [f for f in os.listdir(real_dir) if f.lower().endswith('.jpg')]

print("len",len(jpg_files))