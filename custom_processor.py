import torch
from torchvision.transforms.v2 import functional as F_v2
from transformers.image_utils import is_scaled_image
from transformers.image_processing_utils import BatchFeature
from torchvision.transforms import Normalize

class CustomProcessor:
    def __init__(self,size:int=224,
                 rescale_factor:float=1.0, #0.00392156862745098,
                 image_mean:list=[0.485,0.456,0.406],image_std:list=[0.229,0.224,0.225]):
        self.size=size
        self.rescale_factor=rescale_factor
        self.image_mean=image_mean
        self.image_std=image_std
    
    def __call__(self, images:torch.Tensor)->dict:
        
        #resize
        images=(images+1)/2 #convert from [-1,1] to 0,1
        images=F_v2.resize(images, (self.size,self.size))
        #rescale
        images=self.rescale_factor*images
        #normalize
        images = Normalize(self.image_mean,self.image_std)(images)
        data={"pixel_values":images}
        return data
