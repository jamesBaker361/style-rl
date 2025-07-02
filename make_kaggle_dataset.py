import kagglehub

# Download latest version
path = kagglehub.dataset_download("kaustubhdhote/human-faces-dataset")

print("Path to dataset files:", path)