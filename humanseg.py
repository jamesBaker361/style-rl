from controlnet_aux import UniformerDetector
from PIL import Image

# Load human segmentation preprocessor
seg = UniformerDetector.from_pretrained("lllyasviel/Annotators")

# Load image
image = Image.open("lebrun.jpg")

# Run segmentation
mask = seg(image)

# Save result
mask.save("body_mask.png")


