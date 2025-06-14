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
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torch.nn import functional as F


image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="compare")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    #accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    #with accelerator.autocast():
    generator=torch.Generator(accelerator.device)
    generator.manual_seed(123)

    pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.float16
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
    torch_dtype=torch.float16
    )
    pipeline.enable_vae_tiling()
    pipeline.to(accelerator.device)

    prompt = "a tiny astronaut hatching from an egg on the moon"

    #image1 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256).images[0]
    ip_cross_attention_dim=16
    embedding_dim=4
    intermediate_embedding_dim=8
    
    print("pipleine params",len([p for  name,p in pipeline.transformer.named_parameters()]))

    prepare_ip_adapter(pipeline.transformer,accelerator.device,
                        torch.float16,
                        ip_cross_attention_dim)
    print("pipleine params",len([p for  name,p in pipeline.transformer.named_parameters()]))
    encoder_hid_proj=replace_ip_attn(pipeline.transformer,embedding_dim,intermediate_embedding_dim,ip_cross_attention_dim,4,True,return_encoder_hid_proj=True)
    print("pipleine params",len([p for  name,p in pipeline.transformer.named_parameters()]))
    pipeline.set_encoder_hid_proj(encoder_hid_proj.to(device=accelerator.device,
                                                        dtype=torch.float16
                                                        ))
    print("pipleine params",len([p for  name,p in pipeline.transformer.named_parameters()]))
    '''for block in pipeline.transformer.transformer_blocks:
        print(block.attn2.processor)'''
    #embeds.shape = [N_a,B,N_i,D] N_a= # of adapters, N_i = images per image prompt, D =dimension of embedding
    image1 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=[torch.zeros((1,1,embedding_dim),device=accelerator.device,
                                                                                                                                            dtype=torch.float16
                                                                                                                                            )]).images[0]

    print("b4",len([param for param in pipeline.transformer.parameters() if param.requires_grad])  )# should be False)

    for name,param in pipeline.transformer.named_parameters():
        param.requires_grad_(False)

    attn_layer_list=[m for (n,m) in get_modules_of_types(pipeline.transformer,IPAdapterAttnProcessor2_0)]+[encoder_hid_proj]
    accelerator.print("len attn_layers",len(attn_layer_list))
    for layer in attn_layer_list:
        layer.requires_grad_(True)

    
    print("after",len([param for param in pipeline.transformer.parameters() if param.requires_grad])  )# should be False)

<<<<<<< HEAD
    params=[p for p in pipeline.transformer.parameters() if p.requires_grad]+[p for p in encoder_hid_proj.parameters() if p.requires_grad]
    named_params=[n for (n,p) in pipeline.transformer.named_parameters() if p.requires_grad]+[n for (n,p) in encoder_hid_proj.named_parameters() if p.requires_grad]
    print(named_params)
    optimizer=torch.optim.AdamW(params)
    optimizer,pipeline=accelerator.prepare(optimizer,pipeline)
    output=pipeline.call_with_grad(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=[torch.zeros((1,1,embedding_dim),device=accelerator.device,
                                                                                                                                                        dtype=torch.float16
                                                                                                                                                        )]).images[0]
    target=torch.zeros(output.size(),device=accelerator.device,
                        dtype=torch.float16
                        )
=======
        params=[p for p in pipeline.transformer.parameters() if p.requires_grad]+[p for p in encoder_hid_proj.parameters() if p.requires_grad]
        named_params=[n for (n,p) in pipeline.transformer.named_parameters() if p.requires_grad]+[n for (n,p) in encoder_hid_proj.named_parameters() if p.requires_grad]
        print(named_params)
        optimizer=torch.optim.AdamW(params)
        optimizer,pipeline=accelerator.prepare(optimizer,pipeline)
        output=pipeline.call_with_grad(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=[torch.zeros((1,1,embedding_dim),device=accelerator.device,dtype=torch.float16)],return_dict=False).images[0]
        target=torch.zeros(output.size())
>>>>>>> parent of d843453 (denosie model)

    loss=F.mse_loss(output,target)
    accelerator.backward(loss)

    optimizer.step()
    optimizer.zero_grad()
    

    #pipeline.transformer=replace_ip_attn(pipeline.transformer,ip_cross_attention_dim,512,ip_cross_attention_dim,4,True)
    #image2 = pipeline(prompt=prompt, num_inference_steps=2,generator=generator,height=256,width=256,ip_adapter_image_embeds=torch.zeros((1,1,ip_cross_attention_dim),device=accelerator.device,dtype=torch.bfloat16)).images[0]
    pipeline = CompatibleSanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.float16
    )
    pipeline.enable_vae_tiling()
    pipeline.to(accelerator.device)
    prepare_ip_adapter(pipeline.transformer,accelerator.device,
                        #torch.float16,
                        ip_cross_attention_dim)
    encoder_hid_proj=replace_ip_attn(pipeline.transformer,embedding_dim,intermediate_embedding_dim,ip_cross_attention_dim,4,True,deep_to_ip_layers=True,return_encoder_hid_proj=True)
    pipeline.set_encoder_hid_proj(encoder_hid_proj.to(device=accelerator.device,
                                                        dtype=torch.float16
                                                        ))
    
    image3 = pipeline(prompt=prompt, num_inference_steps=2,height=256,width=256,ip_adapter_image_embeds=[torch.zeros((1,1,embedding_dim),device=accelerator.device,
                                                                                                                        #dtype=torch.float16
                                                                                                                        )]).images[0]

    '''accelerator.log({
        "image1":wandb.Image(image1),
        "image0":wandb.Image(image0),
    # "image2":wandb.Image(image2),
        "image3":wandb.Image(image3),
    })'''



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