import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from embedding_helpers import EmbeddingUtil
from datasets import load_dataset
from torchvision.transforms import ToTensor
import torch

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
    
    src_data=load_dataset(args.src_dataset,split="train")
    for k,row in enumerate(src_data):
        print(row)
        embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(row["image"]))
        for key in ["image","url","tag","champion"]:
            output_dict[key].append(row[key])
        output_dict["embedding"].append(embedding)

        if k >2:
            break


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