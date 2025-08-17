from controlnet_aux import UniformerDetector
from PIL import Image
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector


# Load human segmentation preprocessor
seg = sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

# Load image
image = Image.open("lebrun.jpg")

# Run segmentation
mask = seg(image)

# Save result
mask.save("body_mask.png")


