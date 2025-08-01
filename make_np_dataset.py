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
from sana_pipelines import CompatibleSanaSprintPipeline

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil
from custom_vae import public_encode
from gpu_helpers import find_cuda_objects, delete_unique_objects

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--output_dataset",type=str,default="jlbaker361/captioned-images-npz")
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--rewrite",action="store_true")
parser.add_argument("--text_embedding",action="store_true")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--pipeline",type=str,default="sana",help="sana or lcm")

def main(args):
    random.seed(42)
    with torch.no_grad():
        composition=transforms.Compose([
                transforms.Resize((args.image_size,args.image_size)),
                transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
            ])
        accelerator=Accelerator(mixed_precision=args.mixed_precision)
        with accelerator.autocast():
            device=accelerator.device


            torch_dtype=torch.float16
            try:
                raw_data=load_dataset(args.dataset,split="train")
            except OSError:
                raw_data=load_dataset(args.dataset,split="train",force_download=True)

            raw_data=[row for row in raw_data]
            random.shuffle(raw_data)

            embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

            new_dataset={
                "image":[],
                "embedding":[],
                "text":[],
                "prompt":[],
                "posterior":[]
            }
            if args.pipeline=="sana":
                new_dataset["attention_mask"]=[]

            try:
                old_data=load_dataset(args.output_dataset)
                existing=True
                len_old=len([r for r in old_data])
            except:
                existing=False

            if args.pipeline=="lcm":
                pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
            else:
                pipeline=CompatibleSanaSprintPipeline.from_pretrained("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",device=accelerator.device)
                pipeline.do_classifier_free_guidance=False
            pipeline=pipeline.to(accelerator.device)
            pipeline.text_encoder=pipeline.text_encoder.to(accelerator.device,torch_dtype)
            pipeline.vae=pipeline.vae.to(accelerator.device,torch_dtype)
            pipeline=accelerator.prepare(pipeline)

            before=find_cuda_objects()
            for k,row in enumerate(raw_data):
                print(k)
                if k==args.limit:
                    break
                image=row["image"].convert("RGB").resize((256,256))
                
                text=row["text"]
                prompt=row["text"]
                
                
                
                if args.pipeline=="lcm":
                    encoded_text, _ = pipeline.encode_prompt(
                                                    prompt=text,
                                                    device=accelerator.device,
                                                    num_images_per_prompt=1,
                                                    do_classifier_free_guidance=True
                                            )
                else:
                    encoded_text, encoded_text_attention_mask = pipeline.encode_prompt(
                                                    prompt=text,
                                                    device=accelerator.device,
                                                    num_images_per_prompt=1,
                                            )
                    new_dataset["attention_mask"].append(encoded_text_attention_mask)
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0).cpu().detach().numpy()
                new_dataset["image"].append(image)
                new_dataset["embedding"].append(embedding)
                new_dataset["text"].append(encoded_text)
                
                new_dataset["prompt"].append(prompt)


                image=pipeline.image_processor.preprocess(image)
                
                posterior=public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).squeeze(0).cpu().detach().numpy()
                if k==0:
                    posterior=public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).squeeze(0).cpu().detach().numpy()
                    print("image max min",image.max(),image.min())
                    print("post min max",public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).max(),public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).min())
                    print("posterior",posterior)
                new_dataset["posterior"].append(posterior)
                torch.cuda.empty_cache()
                after=find_cuda_objects()
                delete_unique_objects(before,after)
                if k+1 %500==0:
                    print("processed: ",k+1)
                    if existing==False or args.rewrite:
                        time.sleep(random.randint(1,10))
                        Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)
                    else:
                        if k+1> len_old:
                            time.sleep(random.randint(1,10))
                            Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)




            time.sleep(random.randint(1,10))
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