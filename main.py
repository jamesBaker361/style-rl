import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
from diffusers import DiffusionPipeline
import torch
import time

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="style_creative")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16
    }[args.mixed_precision]


    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
    pipe.to(torch_device="cuda", torch_dtype=torch_dtype)

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
    num_inference_steps = 4
    images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images
    images[0].save("image.png")
    return

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