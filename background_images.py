
real_test_prompt_list=[
           ' in the jungle',
            ' in the snow',
            ' on the beach',
            ' on a cobblestone street',
            ' on top of pink fabric',
            ' on top of a wooden floor',
            ' with a city in the background',
            ' with a mountain in the background',
            ' with a blue house in the background',
            ' on top of a purple rug in a forest',
            ' with a wheat field in the background',
            ' with a tree and autumn leaves in the background',
            ' with the Eiffel Tower in the background',
            ' floating on top of water',
            ' floating in an ocean of milk',
            ' on top of green grass with sunflowers around it',
            ' on top of a mirror',
            ' on top of the sidewalk in a crowded street',
            ' on top of a dirt road',
            ' on top of a white rug',
        ]

import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline
from datasets import load_dataset,Dataset

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")



def main(args):
    accelerator=Accelerator()
    steps=32
    size=1024
    data_dict={
        "image":[],
        "prompt":[]
    }
    
    pipe= StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(accelerator.device)
    for prompt in real_test_prompt_list:
        gen=torch.Generator()
        gen.manual_seed(10101)
        p=prompt.replace("with a"," ").replace("floating on top of", " ").replace("floating in", " ").replace("on top of"," ").replace("with the"," ").replace("in the background", " ")
        image=pipe(p, height=size,width=size,generator=gen).images[0]
        data_dict['image'].append(image)
        data_dict['prompt'].append(prompt)

    Dataset.from_dict(data_dict).push_to_hub("jlbaker361/real_test_prompt_list")

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