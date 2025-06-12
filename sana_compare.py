import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from sana_pipelines import CompatibleSanaSprintPipeline,prepare_ip_adapter
from diffusers import SanaSprintPipeline
import torch
import wandb
from diffusers.utils import load_image
from adapter_helpers import replace_ip_attn

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")

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
    pipeline.enable_vae_tiling()
    pipeline.to(accelerator.device)

    prompt = "a tiny astronaut hatching from an egg on the moon"

    image0 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256).images[0]
    config=pipeline.transformer.config

    generator=torch.Generator(accelerator.device)
    generator.manual_seed(123)

    pipeline = CompatibleSanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
    )
    pipeline.enable_vae_tiling()
    pipeline.to(accelerator.device)

    prompt = "a tiny astronaut hatching from an egg on the moon"

    image1 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256).images[0]
    ip_cross_attention_dim=256
    embedding_dim=512
    
    
    prepare_ip_adapter(pipeline.transformer,accelerator.device,torch.bfloat16,ip_cross_attention_dim)
    encoder_hid_proj=replace_ip_attn(pipeline.transformer,ip_cross_attention_dim,512,ip_cross_attention_dim,4,True,return_encoder_hid_proj=True)
    pipeline.set_encoder_hid_proj(encoder_hid_proj)
    '''for block in pipeline.transformer.transformer_blocks:
        print(block.attn2.processor)'''
    
    image1 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=torch.zeros((1,1,embedding_dim),device=accelerator.device,dtype=torch.bfloat16)).images[0]

    

    #pipeline.transformer=replace_ip_attn(pipeline.transformer,ip_cross_attention_dim,512,ip_cross_attention_dim,4,True)
    #image2 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=torch.zeros((1,1,ip_cross_attention_dim),device=accelerator.device,dtype=torch.bfloat16)).images[0]

    encoder_hid_proj=replace_ip_attn(pipeline.transformer,ip_cross_attention_dim,512,ip_cross_attention_dim,4,True,deep_to_ip_layers=True,return_encoder_hid_proj=True)
    pipeline.set_encoder_hid_proj(encoder_hid_proj)
    image3 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=torch.zeros((1,1,ip_cross_attention_dim),device=accelerator.device,dtype=torch.bfloat16)).images[0]

    accelerator.log({
        "image1":wandb.Image(image1),
        "image0":wandb.Image(image0),
       # "image2":wandb.Image(image2),
        "image3":wandb.Image(image3),
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