import torch
for model_type in ["dino_vits8" ,"dino_vits16" , "dino_vitb8" , "dino_vitb16" , "vit_small_patch8_224" ,"vit_small_patch16_224" , "vit_base_patch8_224" , "vit_base_patch16_224"]:
    model = torch.hub.load('facebookresearch/dino:main', model_type)