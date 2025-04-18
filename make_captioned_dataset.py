from datasets import Dataset,load_dataset
from PIL import Image
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
import time

parser=argparse.ArgumentParser()
parser.add_argument("--src_dataset",type=str,default="timm/imagenet-22k-wds")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/imagenet-captioned")
parser.add_argument("--image_key",type=str,default="jpg")
parser.add_argument("--limit",type=int,default=-1)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype={
        "cuda":torch.float16,
        "cpu":torch.float32
    }[device]

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device,torch_dtype)

    img_list=[]
    text_list=[]
    src_data=load_dataset(args.src_dataset,split="train")
    for i,row in enumerate(src_data):
        if i==args.limit:
            break
        img=row[args.image_key]
        inputs = processor(img, return_tensors="pt").to(device,torch_dtype)

        out = model.generate(**inputs)
        text=processor.decode(out[0], skip_special_tokens=True)

        img_list.append(img)
        text_list.append(text)



    Dataset.from_dict({
        "image":img_list,
        "text":text_list
    }).push_to_hub(args.dest_dataset)

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