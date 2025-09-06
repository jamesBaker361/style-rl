import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
from diffusers import StableDiffusionPipeline
from datasets import Dataset,load_dataset
import time
import random

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--num_inference_steps",type=int,default=16)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--seed_offset",type=int,default=1)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--upload_interval",type=int,default=10)
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/directional")

noun_list = [
"robot",     # machine
"child",  # machine
"woman",    # machine
"adult",    # person
"man"    # person
]

location_list = [
" in a park",
"in a school",
"in a city",
"in the jungle",
"at the beach",
" "
]

action_list=[
        "running",
    "writing",
    "dancing",
    "eating",
    "driving",
    " "
]

style_list = [
",watercolor style",
",pixel art style",
",photorealistic style",
",impressionist style",
",anime style",
",cubist style",
" "
]

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device
    
    pipe = StableDiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(device)

    

    try:
        data_dict=load_dataset(args.dest_dataset,split="train").to_dict()
        start=len(data_dict["image"])
    except:
        data_dict={
        "image":[],
        "noun":[],
        "location":[],
        "action":[],
        "style":[],
        "seed":[],
        "prompt":[]
        }
        start=0



    seed_list=[x+args.seed_offset for x in range(2)]

    k=0
    accelerator.print("starting at ",start)
    for noun in noun_list:
        for location in location_list:
            for action in action_list:
                for style in style_list:
                    for seed in seed_list:
                        if k==args.limit:
                            break
                        else:
                            k+=1
                        if k<=start:
                            continue

                        gen=torch.Generator()
                        gen.manual_seed(seed)
                        prompt=" ".join([noun,location,action,style])
                        accelerator.print(prompt)

                        image=pipe(prompt,height=args.dim,width=args.dim,num_inference_steps=args.num_inference_steps).images[0]

                        data_dict["image"].append(image)
                        data_dict["action"].append(action)
                        data_dict["location"].append(location)
                        data_dict["noun"].append(noun)
                        data_dict["prompt"].append(prompt)
                        data_dict["seed"].append(seed)
                        data_dict["style"].append(style)

                        if k%args.upload_interval==0:
                            try:
                                time.sleep(10+random.randint(1,30))
                                Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)
                            except:
                                time.sleep(10+random.randint(1,30))
                                Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)
    time.sleep(10)
    Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)





    

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