import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import Dataset,load_dataset
import datasets
from image_utils import concat_images_horizontally
import wandb
import matplotlib.pyplot as plt
import PIL
import io

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")
parser.add_argument("--dataset_list",nargs="*",type=str)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--figure_name",type=str,default="fig")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    def get_name(dataset_name):
        if dataset_name.find("attn"):
            return "Monkey"
        
    x=[]
    y=[]
    labels=[]

    for d in args.dataset_list:
        print(d)
        dataset=load_dataset(d,split="train")
        try:
            dataset=dataset.cast_column("image",datasets.Image())
        except:
            pass
        try:
            dataset=dataset.cast_column("augmented_image",datasets.Image())
        except Exception as e:
            print("error!!!")
            print(e)
            pass
        text_score=np.mean(dataset["text_score"])
        dino_score=np.mean(dataset["dino_score"])
        image_score=np.mean(dataset["image_score"])
        x.append(image_score)
        y.append(text_score)
        labels.append(d)
        accelerator.print(f"{d} & {round(text_score,3)} & {round(dino_score,3)} & {round(image_score,3)}  \\\\ ")

    plt.scatter(x,y)
    for xi, yi, label in zip(x, y, labels):
        plt.text(xi, yi, label, fontsize=9, ha='right', va='bottom')

    plt.savefig(f"{args.figure_name}.png")
    for k,rows in enumerate(zip(*[load_dataset(d,split="train") for d in args.dataset_list ])):
        if k==args.limit:
            break
        image_list=rows[0]["image"]+[r["augmented_image"] for r in rows]
        for i in range(len(image_list)):
            image=image_list[i]
            if type(image)==dict:
                image = PIL.Image.open(io.BytesIO(image["bytes"]))
                image_list[i]=image
        print([type(i) for i in image_list])
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