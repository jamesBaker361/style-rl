import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from embedding_helpers import EmbeddingUtil
from datasets import load_dataset,Dataset
import datasets
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import base64
from io import BytesIO

import time

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--src_dataset",type=str,default="jlbaker361/league-dragon-splash-tagged")
parser.add_argument("--embedding",type=str,default="clip",help="clip ssl siglip2 or dino")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/embedding")

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)

    device=accelerator.device


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    output_dict={
        "image":[],
        "url":[],
        "tag":[],
        "champion":[],
        "embedding":[]
    }
    start=0
    try:
        old_dataset=load_dataset(args.dest_dataset,split="train")
        output_dict=old_dataset.to_dict()
        start=len(output_dict["image"])
        print(f"skipping {start}")
    except:
        print("colu,ndt load from hf")
    
    src_data=load_dataset(args.src_dataset,split="train")
    src_data=src_data.cast_column("image",datasets.Image())
    for k,row in enumerate(src_data):
        if k< start:
            continue
        embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(row["image"]))
        print(type(embedding),embedding.size())
        for key in ["url","tag","champion","image"]:
            output_dict[key].append(row[key])
        output_dict["embedding"].append(embedding)

        if k % 50==0:
            Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)


    
    Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)


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