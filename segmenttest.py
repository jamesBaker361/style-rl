import requests
import random

import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from diffusers.utils.loading_utils import load_image

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

#image_url = "https://www.maids.com/wp-content/uploads/2022/12/bigstock-Handsome-Young-Man-Cleaning-Wi-276105073.jpg"
image_url="lebrun.jpg"
image = load_image(image_url)
# Check for cats and remote controls
text_labels = [["head", "arm", "torso","leg","neck","foot"]]

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.2,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

def white_copy_with_rectangles(image: Image.Image, box_list, rect_color_list):
    """
    Make a white copy of an image with rectangles drawn at given coordinates.

    Args:
        image (PIL.Image): The input image.
        boxes (list of tuples): Each tuple is (x1, y1, x2, y2) for rectangle corners.
        rect_color (str or tuple): Color for rectangles (default "black").

    Returns:
        PIL.Image: The new image.
    """
    # Create a white copy
    new_img = Image.new("RGB", image.size, "white")
    draw = ImageDraw.Draw(new_img)

    # Draw rectangles
    for (x1, y1, x2, y2),rect_color in zip(box_list,rect_color_list):
        draw.rectangle([x1, y1, x2, y2], outline=rect_color, fill=rect_color)

    return new_img

result = results[0]
box_list=[]
rect_color_list=[]
for k,(box, score, labels) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
    box = [round(x, 2) for x in box.tolist()] #(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    box_list.append(box)
    rect_color_list.append(["red","blue","green","purple","orange"][k%4])

white_copy=white_copy_with_rectangles(image,box_list,rect_color_list)
new_img=Image.blend(image,white_copy, 0.5)
new_img.save("segmented.png")