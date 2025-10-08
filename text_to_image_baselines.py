import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import load_dataset
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from pipeline_stable_diffusion_3_instruct_pix2pix import StableDiffusion3InstructPix2PixPipeline
from main_seg import real_test_prompt_list
from img_helpers import concat_images_horizontally
import wandb
from eval_helpers import DinoMetric
from transformers import AutoProcessor, CLIPModel
import ImageReward as RM

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--project_name",type=str,default="baseline")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--size",type=int,default=256)

@torch.no_grad()
def main(args):
    ir_model=RM.load("ImageReward-v1.0")
        
        
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    dino_metric=DinoMetric(accelerator.device)

    

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device


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