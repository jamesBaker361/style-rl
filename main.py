import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,AlignPropConfig,AlignPropTrainer
from datasets import load_dataset
import numpy as np
import torch
import time
from PIL import Image
from peft import LoraConfig
from pipelines import KeywordDDPOStableDiffusionPipeline,CompatibleLatentConsistencyModelPipeline
from typing import Any
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
import wandb
from worse_peft import apply_lora
from ml_dtypes import bfloat16

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="style")
parser.add_argument("--prompt",type=str,default="portrait, a beautiful cyborg with golden hair, 8k")
parser.add_argument("--style_dataset",type=str,default="jlbaker361/stylization")
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--limit",type=int,default=5)
parser.add_argument("--method",type=str,default="ddpo")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--style_layers_train",action="store_true",help="only train the style layers")
parser.add_argument("--content_layers_train",action="store_true",help="separately train the style layers")
parser.add_argument("--sample_num_batches_per_epoch",type=int,default=64)
parser.add_argument("--batch_size",type=int,default=2)
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--n_evaluation",type=int,default=10)

def cos_sim_rescaled(vector_i,vector_j,return_np=False):
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    try:
        result= cos(vector_i,vector_j) *0.5 +0.5
    except TypeError:
        result= cos(torch.tensor(vector_i),torch.tensor(vector_j)) *0.5 +0.5
    if return_np:
        return result.detach().cpu().numpy()
    return result

def get_vit_embeddings(vit_processor: ViTImageProcessor, vit_model: BetterViTModel, image_list:list,return_numpy:bool=True):
    '''
    returns (vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list)
    '''
    vit_embedding_list=[]
    vit_content_embedding_list=[]
    vit_style_embedding_list=[]
    for image in image_list:
        do_rescale=True
        if type(image)==torch.Tensor and image.dtype==torch.bfloat16:
            image=image.float()
            do_rescale=False
        vit_inputs = vit_processor(images=[image], return_tensors="pt",do_rescale=do_rescale)
        #print("inputs :)")
        vit_inputs['pixel_values']=vit_inputs['pixel_values'].to(vit_model.device)
        vit_outputs=vit_model(**vit_inputs,output_hidden_states=True, output_past_key_values=True)
        vit_embedding_list.append(vit_outputs.last_hidden_state.reshape(1,-1)[0])
        vit_style_embedding_list.append(vit_outputs.last_hidden_state[0][0]) #CLS token: https://github.com/google/dreambooth/issues/3
        vit_content_embedding_list.append(vit_outputs.past_key_values[11][0].reshape(1,-1)[0])
    if return_numpy:
        vit_embedding_list=[v.cpu().numpy() for v in vit_embedding_list]
        vit_style_embedding_list=[v.cpu().numpy() for v in vit_style_embedding_list]
        vit_content_embedding_list=[v.cpu().numpy() for v in vit_content_embedding_list]
    return vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list

def get_image_logger(keyword:str,accelerator:Accelerator):
    def image_outputs_logger(image_data, global_step, _):
        # For the sake of this example, we will only log the last batch of images
        # and associated data
        result = {}
        images, prompts, _, rewards, _ = image_data[-1]

        for i, image in enumerate(images):
            result[f"{keyword}_{i}"]=wandb.Image(image)

        accelerator.log(
            result,
            step=global_step,
        )
    
    return image_outputs_logger

def set_trainable(sd_pipeline:DiffusionPipeline,keywords:list):
    for key in keywords:
        for name,p in sd_pipeline.unet.named_parameters():
            if name.find(key)!=-1:
                p.requires_grad_(True)

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    with accelerator.autocast():

        def prompt_fn()->tuple[str,Any]:
            return args.prompt, {}

        content_image=Image.open("image.png")

        pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        pipe.to(torch_device="cuda", torch_dtype=torch_dtype)

        prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

        try:
            vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
            vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16').to(accelerator.device)
        except:
        
            vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16',force_download=True)
            vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16',force_download=True).to(accelerator.device)
        vit_model.eval()
        vit_model.requires_grad_(False)

        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        num_inference_steps = args.num_inference_steps
        #images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images
        #images[0].save("image.png")
        data=load_dataset(args.style_dataset,split="train")
        STYLE_LORA="style_lora"
        CONTENT_LORA="content_lora"
        style_score_list=[]
        content_score_list=[]
        for i, row in enumerate(data):
            if i<args.start or i>=args.limit:
                continue
            label=row["label"]
            images=[row[f"image_{k}"] for k in range(4)]
            _,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images+[content_image],False)
            vit_style_embedding_list=vit_style_embedding_list[:-1]
            style_embedding=torch.stack(vit_style_embedding_list).mean(dim=0)
            content_embedding=vit_content_embedding_list[-1]
            evaluation_images=[]
            



                
            ddpo_config=DDPOConfig(log_with="wandb",
                            sample_batch_size=args.batch_size,
                num_epochs=1,
                mixed_precision=args.mixed_precision,
                sample_num_batches_per_epoch=args.sample_num_batches_per_epoch,
                train_batch_size=args.batch_size,
                train_gradient_accumulation_steps=args.gradient_accumulation_steps,
                sample_num_steps=args.num_inference_steps,
                #per_prompt_stat_tracking=True,
                #per_prompt_stat_tracking_buffer_size=32
                )
            align_config=AlignPropConfig(log_with="wandb",num_epochs=1,mixed_precision=args.mixed_precision,
                sample_num_steps=args.num_inference_steps,train_batch_size=args.batch_size,truncated_backprop_timestep=args.num_inference_steps-1,
                truncated_rand_backprop_minmax=[0,args.num_inference_steps])
            sd_pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
            sd_pipeline.unet.to(accelerator.device).requires_grad_(False)
            sd_pipeline.text_encoder.to(accelerator.device).requires_grad_(False)
            sd_pipeline.vae.to(accelerator.device).requires_grad_(False)
            sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae=accelerator.prepare(sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae)
            #sd_pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",device=accelerator.device)
            lora_config=LoraConfig(
                    r=4,
                    lora_alpha=4,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"]
                )
            if args.style_layers_train:

                @torch.no_grad()
                def style_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)-> tuple[list[torch.Tensor],Any]:
                    _,sample_vit_style_embedding_list,__=get_vit_embeddings(vit_processor,vit_model,images,False)
                    #print("len sample",len(sample_vit_style_embedding_list))
                    #print("len images",len(images))
                    if args.method=="align":
                        return torch.cat([cos_sim_rescaled(sample,style_embedding) for sample in sample_vit_style_embedding_list]),{}
                    return [cos_sim_rescaled(sample,style_embedding) for sample in sample_vit_style_embedding_list],{}

                
                style_keywords=[STYLE_LORA]
                sd_pipeline.unet=apply_lora(sd_pipeline.unet,[0],[0],False,keyword=STYLE_LORA)
                style_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,style_keywords)
                print("n trainable layers",len(style_ddpo_pipeline.get_trainable_layers()))
                if args.method=="ddpo":
                    kwargs={"retain_graph":True}
                    style_trainer=BetterDDPOTrainer(
                        ddpo_config,
                        style_reward_function,
                        prompt_fn,
                        style_ddpo_pipeline,
                        get_image_logger(STYLE_LORA+label,accelerator)
                    )
                if args.method=="align":
                    kwargs={}
                    style_trainer=AlignPropTrainer(
                        align_config,
                        style_reward_function,
                        prompt_fn,
                        style_ddpo_pipeline,
                        get_image_logger(STYLE_LORA+label,accelerator)
                        )
            if args.content_layers_train:

                @torch.no_grad()
                def content_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any)->  tuple[list[torch.Tensor],Any]:
                    _,__,sample_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
                    return [cos_sim_rescaled(sample,content_embedding) for sample in sample_vit_content_embedding_list],{}
                
                content_keywords=[CONTENT_LORA]
                sd_pipeline.unet=apply_lora(sd_pipeline.unet,[],[],True,keyword=CONTENT_LORA)
                content_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,[CONTENT_LORA])
                if args.method=="ddpo":
                    kwargs={"retain_graph":True}
                    content_trainer=BetterDDPOTrainer(
                        ddpo_config,
                        content_reward_function,
                        prompt_fn,
                        content_ddpo_pipeline,
                        get_image_logger(CONTENT_LORA+label,accelerator)
                    )
            for e in range(args.epochs):
                if args.style_layers_train:
                    style_trainer.train(**kwargs)
                if args.content_layers_train:
                    content_trainer.train(**kwargs)
            sd_pipeline.unet.requires_grad_(False)
            with torch.no_grad():
                for _ in range(args.n_evaluation):

                    image=sd_pipeline(prompt=args.prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images[0]
                    evaluation_images.append(image)
            

            for image in evaluation_images:
                accelerator.log({f"evaluation_{label}":wandb.Image(image)}) 
            _,evaluation_vit_style_embedding_list,evaluation_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,evaluation_images,False)
            style_score=np.mean([cos_sim_rescaled(sample,style_embedding).cpu() for sample in evaluation_vit_style_embedding_list])
            content_score=np.mean([cos_sim_rescaled(sample, content_embedding).cpu() for sample in evaluation_vit_content_embedding_list])
            accelerator.log({
                f"{label}_content":content_score,
                f"{label}_style":style_score
            })
            content_score_list.append(content_score)
            style_score_list.append(style_score)
        accelerator.log({
            f"content":np.mean(content_score_list),
            f"style":np.mean(style_score_list)
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