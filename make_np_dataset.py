import os
import argparse
from experiment_helpers.gpu_details import print_details
from pipelines import CompatibleLatentConsistencyModelPipeline
from datasets import load_dataset
import torchvision.transforms as transforms

import torch
from accelerate import Accelerator
import time
from datasets import Dataset
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--output_dataset",type=str,default="jlbaker361/captioned-images-npz")
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--rewrite",action="store_true")
parser.add_argument("--text_embedding",action="store_true")

def main(args):
    accelerator=Accelerator()
    device=accelerator.device


    torch_dtype=torch.float32
    try:
        raw_data=load_dataset(args.dataset,split="train")
    except OSError:
        raw_data=load_dataset(args.dataset,split="train",force_download=True)

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    new_dataset={
        "image":[],
        "embedding":[],
        "text":[],
        "prompt":[]
    }

    try:
        old_data=load_dataset(args.output_dataset)
        existing=True
        len_old=len([r for r in old_data])
    except:
        existing=False

    pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)


    for k,row in enumerate(raw_data):
        if k==args.limit:
            break
        image=row["image"].convert("RGB")
        text=row["text"]
        prompt=row["text"]
        text, _ = pipeline.encode_prompt(
                                        text,
                                        "cpu", #accelerator.device,
                                        1,
                                        pipeline.do_classifier_free_guidance,
                                        negative_prompt=None,
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0).cpu().detach().numpy()
        new_dataset["image"].append(image)
        new_dataset["embedding"].append(embedding)
        new_dataset["text"].append(text)
        new_dataset["prompt"].append(prompt)
        if k+1 %500==0:
            if existing==False or args.rewrite:
                time.sleep(random.randint(1,60))
                Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)
            else:
                if k+1> len_old:
                    time.sleep(random.randint(1,60))
                    Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)




    time.sleep(random.randint(1,60))
    Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)
    





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