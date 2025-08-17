
from PIL import Image
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from custom_sam_detector import CustomSamDetector

import os
import warnings
from typing import Union

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# Load human segmentation preprocessor
sam =  SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
custom_sam= CustomSamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

# Load image
image = Image.open("lebrun.jpg")

shape=image.size

masks=custom_sam.get_masks(image)
for ann in masks:
    m = ann['segmentation']
    print(shape,m.shape)
    break

    

# Run segmentation
masked_image = sam(image)

# Save result
masked_image.save("body_mask.png")


print("all done!")