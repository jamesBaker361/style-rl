import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from sana_pipelines import CompatibleSanaSprintPipeline
from diffusers import SanaSprintPipeline
import torch
import wandb

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="compare")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    generator=torch.Generator(accelerator.device)
    generator.manual_seed(123)

    pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
    )
    pipeline.to(accelerator.device)

    prompt = "a tiny astronaut hatching from an egg on the moon"

    image0 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator).images[0]
    

    generator=torch.Generator(accelerator.device)
    generator.manual_seed(123)

    pipeline = CompatibleSanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
    )
    pipeline.to(accelerator.device)

    prompt = "a tiny astronaut hatching from an egg on the moon"

    image1 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator).images[0]

    accelerator.log({
        "image1":wandb.Image(image1),
        "image0":wandb.Image(image0)
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