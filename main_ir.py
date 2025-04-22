import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,AlignPropConfig,AlignPropTrainer
from better_alignprop_trainer import BetterAlignPropTrainer
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import retrieve_timesteps
from datasets import load_dataset
import string
import numpy as np
import torch
import time
from PIL import Image,PngImagePlugin
from pipelines import KeywordDDPOStableDiffusionPipeline,CompatibleLatentConsistencyModelPipeline,PPlusCompatibleLatentConsistencyModelPipeline
from typing import Any
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from trl import DDPOConfig
import wandb
from worse_peft import apply_lora
import torch.nn.functional as F
from gpu_helpers import *
import ImageReward as image_reward
import random
from worse_peft import apply_lora
import pyiqa

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="style")
parser.add_argument("--prompt",type=str,default="portrait, a beautiful cyborg with golden hair, 8k")
parser.add_argument("--style_dataset",type=str,default="jlbaker361/portraits")
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--limit",type=int,default=5)
parser.add_argument("--method",type=str,default="align")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--style_layers_train",action="store_true",help="train the style layers")
parser.add_argument("--content_layers_train",action="store_true",help="separately train the style layers")
parser.add_argument("--style_mid_block",action="store_true")
parser.add_argument("--sample_num_batches_per_epoch",type=int,default=64)
parser.add_argument("--batch_size",type=int,default=2)
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--n_evaluation",type=int,default=10)
parser.add_argument("--style_layers",nargs="*",type=int)
parser.add_argument("--hook_based",action="store_true")
parser.add_argument("--learning_rate",type=float,default=1e-3)
parser.add_argument("--reward_fn",type=str,default="ir",help="ir or qualiclip")
parser.add_argument("--guidance_scale",type=float,default=5.0)
parser.add_argument("--train_whole_model",action="store_true",help="dont use lora")
parser.add_argument("--pretrained_type",type=str,default="consistency",help="consistency or stable")

parser.add_argument("--prompt_src_txt",type=str,default="",help="src of random prompts")
parser.add_argument("--textual_inversion",action="store_true")
parser.add_argument("--placeholder_token",type=str,default="<SKS>")
parser.add_argument("--additional_token",type=str,default="<XYZ>")
parser.add_argument("--num_vectors",type=int,default=1)
parser.add_argument("--initializer_token",type=str,default="pretty")
parser.add_argument("--use_pplus",action="store_true")
parser.add_argument("--validation_epochs",type=int,default=1)
parser.add_argument("--train_unet",action="store_true")



RARE_TOKEN="sksz"

def run_safety_checker(image,*args,**kwargs):
    return image,None

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

def get_image_logger_align(keyword:str,accelerator:Accelerator,cache:list):

    def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
        # For the sake of this example, we will only log the last batch of images
        # and associated data
        result = {}
        images, prompts, _ = [image_pair_data["images"], image_pair_data["prompts"], image_pair_data["rewards"]]
        for i, image in enumerate(images):
            if type(image)==torch.Tensor:
                image=image.float()
            result[f"{keyword}_{i}"]=wandb.Image(image)
            cache.append(image)
        accelerator.log(
            result,
            step=accelerator.get_tracker("wandb").run.step,
        )
    return image_outputs_logger


def mse_reward_fn(*args,**kwargs):
    return -1*F.mse_loss(*args,**kwargs)


def main(args):
    torch.cuda.empty_cache()
    
    if args.style_layers is not None:
        style_layers=[int(n) for n in args.style_layers]
    else:
        style_layers=[]

    if args.textual_inversion:
        args.mixed_precision="no"
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    placeholder_tokens = [args.placeholder_token]

    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    time.sleep(1) #wait a second maybe for accelerator stuff?

    with accelerator.autocast():
        prompt_list=[]
        
        if args.prompt_src_txt!="":
            with open(args.prompt_src_txt, "r") as f:
                prompt_list = [line.strip() for line in f]
        
        if args.reward_fn=="ir":
            ir_normalize=transforms.Normalize(0.16717362830052426,1.0333394966054072)
            ir_model=image_reward.load("ImageReward-v1.0",device=accelerator.device)
            ir_model.to(torch_dtype)
            ir_model=accelerator.prepare(ir_model)
        elif args.reward_fn=="qualiclip":
            qualiclip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            qualiclip_model=torch.hub.load(repo_or_dir="miccunifi/QualiCLIP", source="github", model="QualiCLIP")
            qualiclip_model.to(torch_dtype)
            qualiclip_model=accelerator.prepare(qualiclip_model)


        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        num_inference_steps = args.num_inference_steps

        STYLE_LORA="style_lora"
 
        accelerator.free_memory()
        torch.cuda.empty_cache()
        ddpo_config=DDPOConfig(log_with="wandb",
                                sample_batch_size=args.batch_size,
                                train_learning_rate=args.learning_rate,
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
                                    train_learning_rate=args.learning_rate,
                                    sample_guidance_scale=args.guidance_scale,
                                    log_image_freq=-1,
            sample_num_steps=args.num_inference_steps,train_batch_size=args.batch_size,truncated_backprop_timestep=args.num_inference_steps-1,
            truncated_backprop_rand=False,
            truncated_rand_backprop_minmax=[0,args.num_inference_steps])
        if args.pretrained_type=="consistency":
            sd_pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        elif args.pretrained_type=="stable":
            sd_pipeline=StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5",device=accelerator.device,torch_dtype=torch_dtype)
        if args.use_pplus:
            sd_pipeline=PPlusCompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
            sd_pipeline.to(device=accelerator.device,torch_dtype=torch_dtype)
            sd_pipeline.register_unet()
        sd_pipeline.run_safety_checker=run_safety_checker
        print("before ",sd_pipeline.unet.config.sample_size)
        sd_pipeline.unet.config.sample_size=args.image_size // sd_pipeline.vae_scale_factor
        print("after", sd_pipeline.unet.config.sample_size)
        sd_pipeline.unet.to(accelerator.device)
        sd_pipeline.unet.requires_grad_(False)
        for param in sd_pipeline.unet.parameters():
            param.requires_grad_(False)
        sd_pipeline.text_encoder.to(accelerator.device)
        sd_pipeline.text_encoder.requires_grad_(False)
        sd_pipeline.vae.to(accelerator.device)
        sd_pipeline.vae.requires_grad_(False)
        tokenizer=sd_pipeline.tokenizer
        text_encoder=sd_pipeline.text_encoder


        sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae=accelerator.prepare(sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae)
        layer_agnostic_tokens=[args.placeholder_token]
        if args.textual_inversion:
            sd_pipeline.text_encoder.to(accelerator.device).requires_grad_(True)
            if args.num_vectors < 1:
                raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

            # add dummy tokens for multi-vector
            additional_tokens = []

            if args.num_vectors==1 and args.use_pplus:
                n_layers=sd_pipeline.get_n_layers()
                for j in range(n_layers):
                    additional_tokens.append(f"{args.placeholder_token}_{j}")
            
            suffixes=[''.join(random.choices(string.ascii_letters + string.digits, k=4)) for _ in range(args.num_vectors)]
            for i in range(1, args.num_vectors):
                suffix=suffixes[i]
                if args.use_pplus:
                    n_layers=sd_pipeline.get_n_layers()
                    for j in range(n_layers):
                        additional_tokens.append(f"{args.additional_token}_{suffix}_{j}")
                else:
                    additional_tokens.append(f"{args.additional_token}_{suffix}")
                layer_agnostic_tokens.append(f"{args.additional_token}_{suffix}")
            placeholder_tokens += additional_tokens

            print('placeholder_tokens',placeholder_tokens)
            num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
            print(f"added {num_added_tokens} tokens")

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError(f"The initializer token must be a single token. {args.initializer_token} is {len(token_ids)}")

            initializer_token_id = token_ids[0]
            placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
            #layer_agnostic_token

            # Resize the token embeddings as we are adding new special tokens to the tokenizer
            text_encoder.resize_token_embeddings(len(tokenizer))

            # Initialise the newly added placeholder token with the embeddings of the initializer token
            token_embeds = text_encoder.get_input_embeddings().weight.data
            with torch.no_grad():
                for token_id in placeholder_token_ids:
                    token_embeds[token_id] = token_embeds[initializer_token_id].clone()

            # Freeze all parameters except for the token embeddings in text encoder
            text_encoder.text_model.encoder.requires_grad_(False)
            text_encoder.text_model.final_layer_norm.requires_grad_(False)
            text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
            print("new tokens",placeholder_tokens)
            print("layer_agnostic_tokens",layer_agnostic_tokens)
            if args.use_pplus:
                sd_pipeline.register_new_tokens(layer_agnostic_tokens)

        def prompt_fn()->str:
            if len(prompt_list)>0:
                prompt= random.choice(prompt_list)
            else:
                prompt= args.prompt
            prompt+=" "
            if args.textual_inversion:
                prompt+=" ".join(layer_agnostic_tokens)
            #print("prompt fn returns ",prompt)
            return prompt

        style_cache=[]
        accelerator.free_memory()

        @torch.no_grad()
        def style_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)-> tuple[list[torch.Tensor],Any]:
            if args.reward_fn=="ir":
                text_input=ir_model.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
                prompt_ids_list=text_input.input_ids.to(accelerator.device)
                prompt_attention_mask_list=text_input.attention_mask.to(accelerator.device)
                ret=torch.stack([ ir_model.score_gard(prompt_ids.unsqueeze(0),prompt_attention_mask.unsqueeze(0),
                                                            F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                                                            ) for image,prompt_ids,prompt_attention_mask in zip(images,prompt_ids_list,prompt_attention_mask_list)])
            return ret,{}

        def style_reward_function_align(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)-> tuple[torch.Tensor,Any]:
            new_prompts=[]
            for prompt in prompts:
                #print(f"reward fn prompt before: {prompt}")
                for token in placeholder_tokens+layer_agnostic_tokens:
                    prompt=prompt.replace(token,"")
                new_prompts.append(prompt)
                #print(f"reward fn prompt after {prompt}")
            prompts=new_prompts
            
            #print('prompt_attention_mask_list.size()',prompt_attention_mask_list.size())
            if args.reward_fn=="ir":
                text_input=ir_model.blip.tokenizer(prompts, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
                prompt_ids_list=text_input.input_ids.to(accelerator.device)
                prompt_attention_mask_list=text_input.attention_mask.to(accelerator.device)
                ret=torch.stack([ ir_model.score_gard(prompt_ids.unsqueeze(0),prompt_attention_mask.unsqueeze(0),
                                                            F.interpolate(ir_normalize(image).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                                                            ) for image,prompt_ids,prompt_attention_mask in zip(images,prompt_ids_list,prompt_attention_mask_list)])
            elif args.reward_fn=="qualiclip":
                ret=torch.stack([qualiclip_model(F.interpolate(qualiclip_normalize(image).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)) for image in images])
            return ret,{}
        
        style_keywords=[STYLE_LORA]
        if args.train_unet:
            sd_pipeline.unet=apply_lora(sd_pipeline.unet,style_layers,[0],args.style_mid_block,keyword=STYLE_LORA)
        
        style_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,style_keywords,textual_inversion=args.textual_inversion)
        print("n trainable layers style",len(style_ddpo_pipeline.get_trainable_layers()))
        sd_pipeline.unet.to(accelerator.device)
        kwargs={}
        
        if args.method=="align":
            
            style_trainer=BetterAlignPropTrainer(
                align_config,
                style_reward_function_align,
                prompt_fn,
                style_ddpo_pipeline,
                get_image_logger_align(STYLE_LORA,accelerator,style_cache)
                )
        elif args.method=="ddpo":
            kwargs={"retain_graph":True}
            style_trainer=BetterDDPOTrainer(
                ddpo_config,
                style_reward_function,
                prompt_fn,
                style_ddpo_pipeline,
                get_image_logger(STYLE_LORA,accelerator)
            )
        def get_images_and_scores(n_evaluation:int):  
            #with torch.autocast(accelerator.device,False):  
            evaluation_images=[]
            score_list=[]
            #ir_model.to(torch.float32)
            prompt_list=[]
            with torch.no_grad():
                for k in range(n_evaluation):
                    prompt=prompt_fn()
                    image=sd_pipeline(prompt=[prompt], num_inference_steps=num_inference_steps, guidance_scale=args.guidance_scale,height=args.image_size,width=args.image_size).images[0]
                    evaluation_images.append(image)
                    #print("evaluation before",prompt)
                    for token in placeholder_tokens+layer_agnostic_tokens:
                        prompt=prompt.replace(token,"")
                    #print("evaluation after",prompt)
                    if args.reward_fn=="ir":
                        score_list.append(ir_model.score(prompt,image))
                        prompt_list.append(prompt)
                    elif args.reward_fn=="qualiclip":
                        image=image.resize((224,224))
                        image=transforms.ToTensor()(image)
                        image=qualiclip_normalize(image).unsqueeze(0)
                        image=image.to(accelerator.device,torch_dtype)
                        print("image",image.dtype,image.device)
                        score_list.append(qualiclip_model(image))
                    '''text_input=ir_model.blip.tokenizer([prompt], padding='max_length', truncation=True, max_length=35, return_tensors="pt")
                    prompt_ids_list=text_input.input_ids.to(accelerator.device)
                    prompt_attention_mask_list=text_input.attention_mask.to(accelerator.device)
                    score_list=[ ir_model.score_gard(prompt_ids.unsqueeze(0),prompt_attention_mask.unsqueeze(0),
                                                            F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                                                            ) for image,prompt_ids,prompt_attention_mask in zip(images,prompt_ids_list,prompt_attention_mask_list)]
                    score_list=[s.detach().cpu().numpy().item() for s in score_list]'''
            return evaluation_images,score_list,prompt_list
        
        for model in [sd_pipeline,sd_pipeline.unet, sd_pipeline.vae,sd_pipeline.text_encoder]:
            model.to(accelerator.device)
        total_start=time.time()
        try:
            for e in range(args.epochs):
                accelerator.free_memory()
                start=time.time()
                try:
                    style_trainer.train(**kwargs)
                except torch.cuda.OutOfMemoryError:
                    print("oom epoch ",e)
                    accelerator.free_memory()
                    style_trainer.train(**kwargs)
                accelerator.free_memory()
                
                end=time.time()
                print(f"\t epoch {e} elapsed {end-start}")
                if e%args.validation_epochs==0:
                    evaluation_images,score_list,prompt_list=get_images_and_scores(args.n_evaluation)
                    metrics={"score":np.mean(score_list)}
                    accelerator.log(metrics)
                    #print("promtpt list",prompt_list)
                    for image,prompt in zip(evaluation_images, prompt_list):
                        accelerator.log({prompt.strip(): wandb.Image(image)})
        except  torch.cuda.OutOfMemoryError:
            print(f"FAILED after {e} epochs")
            end=time.time()
        print(f"all epochs elapsed {end-total_start} total steps= {args.epochs} * {args.num_inference_steps} *{args.batch_size}={args.epochs*args.num_inference_steps*args.batch_size}")
        sd_pipeline.unet.requires_grad_(False)
    
    evaluation_images,score_list,prompt_list=get_images_and_scores(50)
    for image,prompt in zip(evaluation_images, prompt_list):
        accelerator.log({prompt.strip(): wandb.Image(image)})
    for image in evaluation_images:
        accelerator.log({f"final_evaluation":wandb.Image(image)})
    for _style_image in style_cache:
        accelerator.log({f"cache_{STYLE_LORA}":wandb.Image(_style_image)})
    
    things_to_free=[sd_pipeline.unet,sd_pipeline.vae,sd_pipeline.text_encoder]
    
    
    for thing in things_to_free:
        accelerator.free_memory(thing)
    torch.cuda.empty_cache()
    metrics={"score":np.mean(score_list),"final_score":np.mean(score_list)}
    accelerator.log(metrics)
    print(metrics)
        

            


                    

    

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