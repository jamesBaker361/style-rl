import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import Dataset,load_dataset
from image_utils import concat_images_horizontally
import wandb

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--dataset_list",nargs="*",type=str)



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    for d in args.dataset_list:
        dataset=load_dataset(d,split="train")
        accelerator.print(f"{d} & {np.mean(dataset["text_score"])} & {np.mean(dataset["dino_score"])} & {np.mean(dataset["image_score"])}  \\\\ ")
    for rows in zip(*[load_dataset(d,split="train") for d in args.dataset_list ]):
        image_list=[rows[0]["image"]]+[r["augmented_image"] for r in rows]
        concat=concat_images_horizontally(image_list)
        accelerator.log({
            "concat":wandb.Image(concat)
        })



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