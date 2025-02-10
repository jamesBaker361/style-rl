import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,AlignPropConfig,AlignPropTrainer
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import retrieve_timesteps
from datasets import load_dataset
import numpy as np
import torch
import time
from PIL import Image,PngImagePlugin
from peft import LoraConfig
from pipelines import KeywordDDPOStableDiffusionPipeline,CompatibleLatentConsistencyModelPipeline
from typing import Any
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
import wandb
from worse_peft import apply_lora
import torch.nn.functional as F
from ml_dtypes import bfloat16
from hook_trainer import HookTrainer
from torchvision import models

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="style")
parser.add_argument("--prompt",type=str,default="portrait, a beautiful cyborg with golden hair, 8k")
parser.add_argument("--style_dataset",type=str,default="jlbaker361/portraits")
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--limit",type=int,default=5)
parser.add_argument("--method",type=str,default="ddpo")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--style_layers_train",action="store_true",help="only train the style layers")
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
parser.add_argument("--reward_fn",type=str,default="cos")
parser.add_argument("--vgg_layer",type=int,default=27)

RARE_TOKEN="sksz"



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
        do_resize=True
        if type(image)==torch.Tensor:
            image=image.to(dtype=vit_model.dtype)
            do_rescale=False
            do_resize=False
            #print("size",image.size())
            image=F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            vit_inputs={
                "pixel_values":image
            }
        elif type(image)==PngImagePlugin.PngImageFile:
            image=image.convert("RGB")
            try:
                vit_inputs = vit_processor(images=[image], return_tensors="pt",do_rescale=do_rescale,do_resize=do_resize)
            except ValueError as e:
                print("type image",type(image))
                raise
        else:
            try:
                vit_inputs = vit_processor(images=[image], return_tensors="pt",do_rescale=do_rescale,do_resize=do_resize)
            except ValueError as e:
                print("type image",type(image))
                raise
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

def get_image_logger_align(keyword:str,accelerator:Accelerator):

    def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
        # For the sake of this example, we will only log the last batch of images
        # and associated data
        result = {}
        images, prompts, _ = [image_pair_data["images"], image_pair_data["prompts"], image_pair_data["rewards"]]
        for i, image in enumerate(images):
            result[f"{keyword}_{i}"]=wandb.Image(image)

        accelerator.log(
            result,
            step=accelerator.get_tracker("wandb").run.step,
        )
    return image_outputs_logger

def set_trainable(sd_pipeline:DiffusionPipeline,keywords:list):
    for key in keywords:
        for name,p in sd_pipeline.unet.named_parameters():
            if name.find(key)!=-1:
                p.requires_grad_(True)


vgg_image_transforms = transforms.Compose(
        [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )




def main(args):
    torch.cuda.empty_cache()

    image_transforms = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    if args.style_layers is not None:
        style_layers=[int(n) for n in args.style_layers]
    else:
        style_layers=[0]
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    time.sleep(1) #wait a second maybe for accelerator stuff?

    with accelerator.autocast():

        def prompt_fn()->tuple[str,Any]:
            return args.prompt, {}
        
        vgg_extractor=models.vgg16(pretrained=True).features[:args.vgg_layer].eval().to(device=accelerator.device,dtype=torch_dtype)
        vgg_extractor.requires_grad_(False)
        vgg_extractor=accelerator.prepare(vgg_extractor)
        
        def get_vgg_embedding(vgg_extractor:torch.nn.modules.container.Sequential, image:torch.Tensor)->torch.Tensor:
            if type(image)!=torch.Tensor:
                image=transforms.ToTensor()(image)
            
            image=image.to(dtype=torch_dtype)
            image=image.to(device=accelerator.device)
            image.requires_grad_(True)
            image=vgg_image_transforms(image)
            #image=image.float()
            image=F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            return vgg_extractor(image)

        

        pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
        # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
        pipe.to(torch_device="cuda", torch_dtype=torch_dtype)
        hooks = []
        if args.method=="hook":

            content_target_activations = {}

            # Hook function
            def hook_fn(module, input, output):
                content_target_activations[module] = output

            # Register hooks for all encoder and decoder blocks
            blocks=[block for i,block in enumerate(pipe.unet.down_blocks) if i in style_layers]
            if args.style_mid_block:
                blocks.append(pipe.unet.mid_block)
            
            for layer in blocks:
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)  # Keep track of hooks for later removal

            print(f"Registered {len(hooks)} hooks.")

        content_image=pipe(prompt=args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images[0]
        print("content_image type",type(content_image))
        for hook in hooks:
            hook.remove()

        accelerator.log({
            "src_content_image":wandb.Image(content_image)
        })

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
            hooks=[]
            accelerator.free_memory()
            if i<args.start or i>=args.limit:
                continue
            label=row["label"]
            n_image=4
            try:
                images=[row[f"image_{k}"] for k in range(n_image)]
            except:
                n_image=1
                images=[row[f"image_{k}"] for k in range(n_image)]

            _,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images+[content_image],False)
            vit_style_embedding_list=vit_style_embedding_list[:-1]
            style_embedding=torch.stack(vit_style_embedding_list).mean(dim=0)
            vgg_style_embedding=torch.stack([get_vgg_embedding(vgg_extractor,image) for image in images]).mean(dim=0)

            content_embedding=vit_content_embedding_list[-1]
            evaluation_images=[]


                
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
                sample_num_steps=args.num_inference_steps,train_batch_size=args.batch_size,truncated_backprop_timestep=args.num_inference_steps-1,
                truncated_rand_backprop_minmax=[0,args.num_inference_steps])
            sd_pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
            sd_pipeline.unet.to(accelerator.device).requires_grad_(False)
            sd_pipeline.text_encoder.to(accelerator.device).requires_grad_(False)
            sd_pipeline.vae.to(accelerator.device).requires_grad_(False)

            if args.method=="hook":
                _style_target_activations={}

                def style_hook_fn(module, input, output):
                    output[0].requires_grad_(True)
                    if module not in _style_target_activations:
                        _style_target_activations[module]=[]
                    _style_target_activations[module].append(output[0])

                style_blocks=[block for i,block in enumerate(sd_pipeline.unet.down_blocks) if i in style_layers]
                if args.style_mid_block:
                    style_blocks.append(sd_pipeline.unet.mid_block)
                
                for layer in style_blocks:
                    hook = layer.register_forward_hook(style_hook_fn)
                    hooks.append(hook)  # Keep track of hooks for later removal

                print(f"\tRegistered {len(hooks)} hooks.")

                for image in images:
                    timesteps, num_inference_steps = retrieve_timesteps(
                        sd_pipeline.scheduler, num_inference_steps, accelerator.device, None, original_inference_steps=None)
                    timesteps=timesteps[-1].unsqueeze(0).cpu()
                    pixels=image_transforms(image).unsqueeze(0).to(device=accelerator.device,dtype=torch_dtype)
                    model_input = sd_pipeline.vae.encode(pixels).latent_dist.sample()
                    model_input = model_input * sd_pipeline.vae.config.scaling_factor
                    noise = torch.randn_like(model_input)
                    noisy_model_input = sd_pipeline.scheduler.add_noise(model_input, noise, timesteps)

                    sd_pipeline(" ",timesteps=timesteps,latents=noisy_model_input,height=args.image_size,width=args.image_size)

                for hook in hooks:
                    hook.remove()
                hooks=[]
                print("len  _style_target_activations",len(_style_target_activations))
                for k,v in _style_target_activations.items():
                    print("len values",len(v))
                    break
                style_target_activations={}
                for k,v in _style_target_activations.items():
                    style_target_activations[k]=torch.stack([v[i] for i in range(n_image-1, len(v), n_image)]).mean(dim=0)


                    

            sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae=accelerator.prepare(sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae)

            if args.style_layers_train:

                def mse_reward_fn(*args,**kwargs):
                    return -1*F.mse_loss(*args,**kwargs)

                @torch.no_grad()
                def style_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)-> tuple[list[torch.Tensor],Any]:
                    if args.reward_fn=="cos" or args.reward_fn=="mse":
                        _,sample_vit_style_embedding_list,__=get_vit_embeddings(vit_processor,vit_model,images,False)
                        if args.reward_fn=="mse":
                            reward_fn=mse_reward_fn
                        elif args.reward_fn=="cos":
                            reward_fn=cos_sim_rescaled
                        return [reward_fn(sample,style_embedding) for sample in sample_vit_style_embedding_list],{}
                    elif args.reward_fn=="vgg":
                        sample_embedding_list=[get_vgg_embedding(vgg_extractor,image) for image in images]
                        return [mse_reward_fn(sample,vgg_style_embedding,reduction="mean") for sample in sample_embedding_list],{}
                
                def style_reward_function_align(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)-> tuple[torch.Tensor,Any]:
                    if args.reward_fn=="cos" or args.reward_fn=="mse":
                        _,sample_vit_style_embedding_list,__=get_vit_embeddings(vit_processor,vit_model,images,False)
                        if args.reward_fn=="mse":
                            reward_fn=mse_reward_fn
                        elif args.reward_fn=="cos":
                            reward_fn=cos_sim_rescaled
                        return torch.stack([reward_fn(sample,style_embedding) for sample in sample_vit_style_embedding_list]),{}
                    elif args.reward_fn=="vgg":
                        sample_embedding_list=[get_vgg_embedding(vgg_extractor,image) for image in images]
                        
                        return torch.stack([mse_reward_fn(sample,vgg_style_embedding,reduction="mean") for sample in sample_embedding_list]),{}

                
                style_keywords=[STYLE_LORA]
                sd_pipeline.unet=apply_lora(sd_pipeline.unet,style_layers,[0],args.style_mid_block,keyword=STYLE_LORA)
                
                style_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,style_keywords)
                print("n trainable layers style",len(style_ddpo_pipeline.get_trainable_layers()))
                sd_pipeline.unet.to(accelerator.device)
                kwargs={}
                if args.method=="ddpo":
                    kwargs={"retain_graph":True}
                    style_trainer=BetterDDPOTrainer(
                        ddpo_config,
                        style_reward_function,
                        prompt_fn,
                        style_ddpo_pipeline,
                        get_image_logger(STYLE_LORA+label,accelerator)
                    )
                elif args.method=="align":
                    
                    style_trainer=AlignPropTrainer(
                        align_config,
                        style_reward_function_align,
                        prompt_fn,
                        style_ddpo_pipeline,
                        get_image_logger_align(STYLE_LORA+label,accelerator)
                        )
                elif args.method=="hook":
                    style_trainer=HookTrainer(
                        accelerator,
                        args.epochs,
                        args.num_inference_steps,
                        args.gradient_accumulation_steps,
                        args.sample_num_batches_per_epoch,
                        style_ddpo_pipeline,
                        prompt_fn,
                        args.image_size,
                        style_target_activations,
                        label,
                        train_learning_rate=args.learning_rate
                    )
                if args.reward_fn=="vgg":
                    kwargs={"retain_graph":True}
            if args.content_layers_train:


                @torch.no_grad()
                def content_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any)->  tuple[list[torch.Tensor],Any]:
                    _,__,sample_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
                    return [cos_sim_rescaled(sample,content_embedding) for sample in sample_vit_content_embedding_list],{}
                
                def content_reward_function_align(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)->tuple[torch.Tensor,Any]:
                    _,__,sample_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
                    return torch.stack([cos_sim_rescaled(sample,content_embedding) for sample in sample_vit_content_embedding_list]),{}
                
                content_keywords=[CONTENT_LORA]
                sd_pipeline.unet=apply_lora(sd_pipeline.unet,[],[],True,keyword=CONTENT_LORA)
                content_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,[CONTENT_LORA])
                print("n trainable layers content",len(content_ddpo_pipeline.get_trainable_layers()))
                if args.method=="ddpo":
                    kwargs={"retain_graph":True}
                    content_trainer=BetterDDPOTrainer(
                        ddpo_config,
                        content_reward_function,
                        prompt_fn,
                        content_ddpo_pipeline,
                        get_image_logger(CONTENT_LORA+label,accelerator)
                    )
                elif args.method=="align":
                    content_trainer=AlignPropTrainer(
                        align_config,
                        content_reward_function_align,
                        prompt_fn,
                        content_ddpo_pipeline,
                        None
                    )
            for e in range(args.epochs):
                torch.cuda.empty_cache()
                start=time.time()
                if args.style_layers_train:
                    style_trainer.train(**kwargs)
                    accelerator.free_memory()
                if args.content_layers_train:
                    content_trainer.train(**kwargs)
                    accelerator.free_memory()
                end=time.time()
                print(f"\t {label} epoch {e} elapsed {end-start}")
            sd_pipeline.unet.requires_grad_(False)
            for hook in hooks:
                hook.remove() 
            with torch.no_grad():
                for _ in range(args.n_evaluation):

                    image=sd_pipeline(prompt=args.prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images[0]
                    evaluation_images.append(image)
                    print("evaluation image of type",type(image))
            

            for image in evaluation_images:
                accelerator.log({f"evaluation_{label}":wandb.Image(image)}) 
            _,evaluation_vit_style_embedding_list,evaluation_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,evaluation_images,False)
            style_score=np.mean([cos_sim_rescaled(sample,style_embedding).cpu() for sample in evaluation_vit_style_embedding_list])
            content_score=np.mean([cos_sim_rescaled(sample, content_embedding).cpu() for sample in evaluation_vit_content_embedding_list])
            metrics={
                f"{label}_content":content_score,
                f"{label}_style":style_score
            }
            accelerator.log(metrics)
            content_score_list.append(content_score)
            style_score_list.append(style_score)
            sd_pipeline.unet,sd_pipeline= accelerator.free_memory(sd_pipeline.unet,sd_pipeline)
            if args.style_layers_train:
                style_trainer,style_ddpo_pipeline=accelerator.free_memory(style_trainer,style_ddpo_pipeline)
            if args.content_layers_train:
                content_trainer,content_ddpo_pipeline=accelerator.free_memory(content_trainer,content_ddpo_pipeline)
        metrics={
            f"content":np.mean(content_score_list),
            f"style":np.mean(style_score_list)
            }
        accelerator.log(metrics)
        

            


                    

    

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