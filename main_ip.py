import os
import sys
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time

sys.path.append(os.path.dirname(__file__))
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey,insert_monkey, set_ip_adapter_scale_monkey
import torch
from image_utils import concat_images_horizontally
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image
import random
from transformers import AutoProcessor, CLIPModel
from pipelines import CompatibleLatentConsistencyModelPipeline
#import ImageReward as RM
from eval_helpers import DinoMetric

import datasets
from datasets import Dataset
import wandb
import numpy as np
from prompt_list import real_test_prompt_list

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/ssl-league_captioned_splash-1000-sana")
parser.add_argument("--initial_steps",type=int,default=8,help="how many steps for the inference")
parser.add_argument("--limit",type=int,default=-1,help="limit of samples")
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--scale",type=float,default=0.75)
parser.add_argument("--dest_dataset",type=str, default="jlbaker361/monkey")
parser.add_argument("--object",type=str,default="character")


def main(args):
    with torch.no_grad():
        #ir_model=RM.load("ImageReward-v1.0")
        
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

        dino_metric=DinoMetric(accelerator.device)

        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)

        # Load IP-Adapter
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

        setattr(pipe,"safety_checker",None)

        

        #monkey_attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)
        try:
            data=datasets.load_dataset(args.src_dataset)
        except:
            data=datasets.load_dataset(args.src_dataset,download_mode="force_redownload")
        data=data["train"]


        output_dict={
        "image":[],
        "augmented_image":[],
        "text_score":[],
        "image_score":[],
        "dino_score":[],
        "prompt":[]
        }

        for k,row in enumerate(data):
            if k==args.limit:
                break
            reset_monkey(pipe)
            ip_adapter_image=row["image"]
            object=args.object
            if "object" in row:
                object=row["object"]
            prompt=object+real_test_prompt_list[k % len(real_test_prompt_list)]

            generator=torch.Generator()
            generator.manual_seed(123)
            pipe.set_ip_adapter_scale(args.scale)
            initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

            inputs = processor(
                text=[prompt], images=[ip_adapter_image,initial_image], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)
            
            #logits_per_text = outputs.logits_per_text.numpy()[0]  # this is the image-text similarity score
            image_embeds=outputs.image_embeds
            text_embeds=outputs.text_embeds
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
            #accelerator.print("logits",logits_per_text.size())

            image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]
            [_,text_score]=logits_per_text
            [_,image_score]=image_similarities
            #[ir_score_normal,ir_score_unmasked, ir_score_seg_mask, ir_score_raw_mask,ir_score_all_steps]=ir_model.score(prompt,[final_image_normal,final_image_unmasked,final_image_seg_mask,final_image_raw_mask,final_image_all_steps])
            [dino_score]=dino_metric.get_scores(ip_adapter_image, [initial_image])

            

            
            

            output_dict["augmented_image"].append(initial_image)
            output_dict["image"].append(ip_adapter_image)
            output_dict["dino_score"].append(dino_score)
            output_dict["image_score"].append(image_score)
            output_dict["text_score"].append(text_score)
            output_dict["prompt"].append(prompt)

            

        Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)
        for key in ["image_score","dino_score","text_score"]:
            accelerator.print(key, np.mean(output_dict[key]))
    

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