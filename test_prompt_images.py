real_test_prompt_list=[
           ' in the jungle',
            ' in the snow',
            ' on the beach',
            ' on a cobblestone street',
            ' on top of pink fabric',
            ' on top of a wooden floor',
            ' with a city in the background',
            ' with a mountain in the background',
            ' with a blue house in the background',
            ' on top of a purple rug in a forest',
            ' with a wheat field in the background',
            ' with a tree and autumn leaves in the background',
            ' with the Eiffel Tower in the background',
            ' floating on top of water',
            ' floating in an ocean of milk',
            ' on top of green grass with sunflowers around it',
            ' on top of a mirror',
            ' on top of the sidewalk in a crowded street',
            ' on top of a dirt road',
            ' on top of a white rug',
        ]

import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from gpu_helpers import *
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from datasets import Dataset

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="test_prompt_pics")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/comparison_images",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--images_per_prompt",type=int,default=4)
parser.add_argument("--num_inference_steps",type=int,default=8)
parser.add_argument("--dim",type=int,default=256)

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
         accelerator.init_trackers(project_name=args.project_name,config=vars(args))





    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    pipe = DiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch_dtype,
    ).to("cuda")

    data_dict={"prompt":[]}
    for k in range(args.images_per_prompt):
        data_dict[f"{k}_null"]=[]
        data_dict[f"{k}_prompt"]=[]

    for p,prompt in enumerate(real_test_prompt_list):
        data_dict['prompt'].append(prompt)
        for k in range(args.images_per_prompt):
            generator=torch.Generator()
            generator.manual_seed(k)
            prompt_image=pipe(prompt,height=args.dim,width=args.dim,num_inference_steps=args.num_inference_steps,generator=generator).images[0]
            null_image=pipe(" ",height=args.dim,width=args.dim,num_inference_steps=args.num_inference_steps,generator=generator).images[0]
            data_dict[f"{k}_null"].append(null_image)
            data_dict[f"{k}_prompt"].append(prompt_image)

    Dataset.from_dict(data_dict).push_to_hub(args.name)








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