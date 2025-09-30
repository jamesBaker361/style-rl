# compare these to ip + background

import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import load_dataset,Dataset
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from pipeline_stable_diffusion_3_instruct_pix2pix import StableDiffusion3InstructPix2PixPipeline
from main_seg import real_test_prompt_list
from img_helpers import concat_images_horizontally
import wandb
from eval_helpers import DinoMetric
from transformers import AutoProcessor, CLIPModel
#import ImageReward as RM

parser=argparse.ArgumentParser()

model_list=[
    "pix2pix","instruct_clip","ultra_edit",
]

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--dest_dataset",type=str, default="jlbaker361/instruct")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--project_name",type=str,default="baseline")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--model",default="pix2pix",help=f"one of {model_list}")
parser.add_argument("--size",type=int,default=256)
parser.add_argument("--background",action="store_true")
parser.add_argument("--object",type=str,default="character")


@torch.no_grad()
def main(args):
    #ir_model=RM.load("ImageReward-v1.0")
        
        
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

    model_id = "timbrooks/instruct-pix2pix"
    
    if args.model=="pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif args.model=="instruct_clip":
        pipe.load_lora_weights("SherryXTChen/InstructCLIP-InstructPix2Pix")
    
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    elif args.model=="ultra_edit":
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_freeform", torch_dtype=torch_dtype)
    pipe.to(device)
    data=load_dataset(args.src_dataset, split="train")

    background_data=load_dataset("jlbaker361/real_test_prompt_list",split="train")
    background_dict={row["prompt"]:row["image"] for row in background_data}

    text_score_list=[]
    image_score_list=[]
    image_score_background_list=[]
    ir_score_list=[]
    dino_score_list=[]

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
        
        prompt=real_test_prompt_list[k%len(real_test_prompt_list)]
        background_image=background_dict[prompt]
        image=row["image"].resize((args.size,args.size))

        object=args.object
        if "object" in row:
            object=row["object"]

        augmented_image=pipe(prompt=object+" "+prompt,height=args.size,width=args.size,image=image,num_inference_steps=args.num_inference_steps).images[0]

        concat=concat_images_horizontally([image,augmented_image])

        accelerator.log({
            f"image_{k}":wandb.Image(concat)
        })

        inputs = processor(
                text=[prompt], images=[image,augmented_image,background_image], return_tensors="pt", padding=True
        )

        outputs = clip_model(**inputs)
        image_embeds=outputs.image_embeds
        text_embeds=outputs.text_embeds
        logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
        #accelerator.print("logits",logits_per_text.size())

        image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]

        [_,text_score,__]=logits_per_text
        [_,image_score,image_score_background]=image_similarities
        #ir_score=ir_model.score(prompt,augmented_image)
        dino_score=dino_metric.get_scores(image, [augmented_image])

        text_score_list.append(text_score.detach().cpu().numpy())
        image_score_list.append(image_score)
        image_score_background_list.append(image_score_background)
        #ir_score_list.append(ir_score)
        dino_score_list.append(dino_score)

        output_dict["augmented_image"].append(augmented_image)
        output_dict["image"].append(image)
        output_dict["dino_score"].append(dino_score)
        output_dict["image_score"].append(image_score)
        output_dict["text_score"].append(text_score)
        output_dict["prompt"].append(prompt)

    accelerator.log({
        "text_score_list":np.mean(text_score_list),
        "image_score_list":np.mean(image_score_list),
        "image_score_background_list":np.mean(image_score_background_list),
       # "ir_score_list":np.mean(ir_score_list),
        "dino_score_list":np.mean(dino_score_list)
    })

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