from datasets import Dataset
import os
from PIL import Image

path= os.path.join("samples","sample")
data_dict={"image":[]}

for file in os.listdir(path):
    image=Image.open(os.path.join(path, file))
    data_dict["image"].append(image)

Dataset.from_dict(data_dict).push_to_hub("jlbaker361/sshq-100")