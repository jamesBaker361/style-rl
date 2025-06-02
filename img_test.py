from PIL import Image
from datasets import load_dataset
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

def concat_images_horizontally(images):
    """
    Concatenate a list of PIL.Image objects horizontally.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images concatenated side-by-side.
    """
    # Resize all images to the same height (optional)
    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [
        img if img.height == min_height else img.resize(
            (int(img.width * min_height / img.height), min_height),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total width and max height
    total_width = sum(img.width for img in resized_images)
    height = min_height

    # Create new blank image
    new_img = Image.new('RGB', (total_width, height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

data=load_dataset("jlbaker361/dino-celeb_captioned-20",split="train")
for row in data:
    break




image=row["image"]

processed=pipe.image_processor.preprocess(image)
reconstructed=pipe.image_processor.postprocess(processed)[0]

img_20=pipe("woman",ip_adapter_image=image,num_inference_steps=20,height=256,width=256).images[0]
img_10=pipe("woman",ip_adapter_image=image,num_inference_steps=10,height=256,width=256).images[0]
img_4=pipe("woman",ip_adapter_image=image,num_inference_steps=4,height=256,width=256).images[0]
img_2=pipe("woman",ip_adapter_image=image,num_inference_steps=2,height=256,width=256).images[0]

concat=concat_images_horizontally([image,reconstructed,img_20,img_10,img_4,img_2])
concat.save("comparison.jpg")