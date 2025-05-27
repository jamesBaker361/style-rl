import torch
import datasets
from pipelines import CompatibleLatentConsistencyModelPipeline
from transformers import CLIPModel, AutoProcessor

for model_type in ["dino_vits8" ,"dino_vits16" , "dino_vitb8" , "dino_vitb16" , "vit_small_patch8_224" ,"vit_small_patch16_224" , "vit_base_patch8_224" , "vit_base_patch16_224"]:
    try:
        model = torch.hub.load('facebookresearch/dino:main', model_type)
    except:
        print("coouldnt find ",model_type)

for data in ["league_captioned_tile","league_captioned_splash","coco_captioned","art_coco_captioned","celeb_captioned"]:
    path=f"jlbaker361/{data}"
    datasets.load_dataset(path)

pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")