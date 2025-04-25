from PIL import Image
import requests
from transformers import AutoProcessor, SiglipModel
import torch

model = SiglipModel.from_pretrained("google/siglip2-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

candidate_labels = ["2 cats", "2 dogs"]
texts = [f"This is a photo of {label}." for label in candidate_labels]

inputs = processor(text=texts, images=image, padding="max_length", max_length=64, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")