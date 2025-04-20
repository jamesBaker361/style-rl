import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from datasets import load_dataset
import torchvision.transforms as transforms
import torch
from accelerate import Accelerator
import time
from diffusers.models.embeddings import IPAdapterFullImageProjection
from extractor import ViTExtractor
import torch.nn.functional as F
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--epochs",type=int,default=1)
parser.add_argument("--embedding",type=str,default="dino")
parser.add_argument("--facet",type=str,default="token",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--data_dir",type=str,default="data_dir")
parser.add_argument("--save_data_npz",action="store_true")
parser.add_argument("--load_data_npz",action="store_true")

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    raw_data=load_dataset(args.dataset,split="train")

    os.makedirs(args.data_dir,exist_ok=True)

    if args.embedding=="dino":
        dino_vit_extractor=ViTExtractor("vit_base_patch16_224",device=accelerator.device)
        dino_vit_extractor.model.eval()
        dino_vit_extractor.model.requires_grad_(False)

    def embed_img_tensor(img_tensor:torch.Tensor)->torch.Tensor:
        if args.embedding=="dino":
            if len(img_tensor.size())==3:
                img_tensor=img_tensor.unsqueeze(0)
            img_tensor=F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            #dino_vit_prepocessed=dino_vit_extractor.preprocess_pil(content_image.resize((args.image_size,args.image_size))).to(dtype=torch_dtype,device=accelerator.device)
            dino_vit_features=dino_vit_extractor.extract_descriptors(img_tensor,facet=args.facet)
            batch_size=img_tensor.size()[0]
            embedding=dino_vit_features.view(batch_size,-1)
        return embedding
    
    def transform_image(pil_image:Image.Image):
        if args.embedding=="dino":
            t=transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize(dino_vit_extractor.mean,dino_vit_extractor.std)]
            )
        return t(pil_image)
    
    embedding_list=[]
    text_list=[]
    for row in raw_data:
        image=row["image"]
        text=row["text"]
        embedding_list.append(embed_img_tensor(transform_image(image)))
        text_list.append(text)




    #the output of the embeddign thing can be passed as ip_adapter_image_embeds or the image itself can be passed as     ip_adapter_image to the pipeline


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")