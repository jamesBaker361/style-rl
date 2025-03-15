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
import numpy as np
import torch
import time
from PIL import Image,PngImagePlugin
from pipelines import KeywordDDPOStableDiffusionPipeline,CompatibleLatentConsistencyModelPipeline
from typing import Any
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from trl import DDPOConfig
import wandb
from worse_peft import apply_lora
import torch.nn.functional as F
from torchvision import models
from statics import unformatted_prompt_list
from facenet_pytorch import MTCNN, InceptionResnetV1
from better_mtcnn import BetterMTCNN
from typing import Union
from diffusers.models.embeddings import ImageProjection
from extractor import ViTExtractor
from dift_sd import SDFeaturizer
from image_projection import PromptImageProjection

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
parser.add_argument("--reward_fn",type=str,default="cos")
parser.add_argument("--content_reward_fn",type=str,default="mse",help="mse or face or vae or raw or dino or dift")
parser.add_argument("--vgg_layer_style",type=int,default=27)
parser.add_argument("--guidance_scale",type=float,default=5.0)
parser.add_argument("--train_whole_model",action="store_true",help="dont use lora")
parser.add_argument("--pretrained_type",type=str,default="consistency",help="consistency or stable")
parser.add_argument("--use_unformatted_prompts",action="store_true")
parser.add_argument("--content_dataset",type=str,default="jlbaker361/people")
parser.add_argument("--content_start",type=int,default=0)
parser.add_argument("--content_limit",type=int,default=5)
parser.add_argument("--content_mid_block",action="store_true")
parser.add_argument("--content_layers",nargs="*",type=int)
parser.add_argument("--prompt_embedding_conditioning",action="store_true")
parser.add_argument("--adapter_conditioning",action="store_true")
parser.add_argument("--num_image_text_embeds",type=int,default=32,help="num_image_text_embeds for image projection")
parser.add_argument("--image_embeds_type",type=str,default="face",help="face or vit, what model to use for the image embeds")
parser.add_argument("--use_encoder_hid_proj",action="store_true",help="whether to use encoder hidden proj thing")
parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map for dift')
parser.add_argument('--t', default=261, type=int, 
                        help='time step for diffusion, choose from range [0, 1000] for dift')
parser.add_argument('--ensemble_size', default=1, type=int, 
                        help='number of repeated images in each batch used to get features for dift')
parser.add_argument("--facet",type=str,default="token",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")


RARE_TOKEN="sksz"

def run_safety_checker(image,*args,**kwargs):
    return image,None

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
            image.requires_grad_(True)
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
        #print("vit_outputs.past_key_values",len(vit_outputs.past_key_values))
        #print("vit_outputs.past_key_values[11]",len(vit_outputs.past_key_values[11]))
        #print("vit_outputs.past_key_values[11][0]",vit_outputs.past_key_values[11][0].size())
        #print("vit_outputs.past_key_values[11][0].reshape(1,-1)",vit_outputs.past_key_values[11][0].reshape(1,-1).size())
        content=vit_outputs.past_key_values[11][0].reshape(1,-1)
        vit_content_embedding_list.append(content)
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

def mse_reward_fn(*args,**kwargs):
    return -1*F.mse_loss(*args,**kwargs)


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
        style_layers=[]
    if args.content_layers is not None:
        content_layers=[int(n) for n in args.content_layers]
    else:
        content_layers=[]
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


        def get_face_embeddings(image:Union[Image.Image, torch.Tensor],resnet:InceptionResnetV1, mtcnn:BetterMTCNN,grad:bool=True)-> torch.Tensor:
        
            if type(image)==torch.Tensor:
                assert len(image.size())==3
                image=255*image.permute( 1, 2, 0)
                #print("line 136 type,device", image.dtype, image.device)
            img_cropped = mtcnn(image)
            '''if type(img_cropped)==torch.Tensor:
                print("line 138 type,device", img_cropped.dtype, img_cropped.device)'''
            if img_cropped is None:  # Handle case where no face is detected
                img_cropped = torch.zeros((3, 160, 160), dtype=torch_dtype, device=accelerator.device)
                img_cropped.requires_grad_(True)
                print("no face !!!")
            #img_cropped.requires_grad_(grad)
            img_embedding=resnet(img_cropped.unsqueeze(0))
            #print("img embedding shape",img_embedding.size())
            return img_embedding

        def prompt_fn()->tuple[str,Any]:
            return args.prompt, {}
        
        vgg_extractor_style=models.vgg16(pretrained=True).features[:args.vgg_layer_style].eval().to(device=accelerator.device,dtype=torch_dtype)
        vgg_extractor_style.requires_grad_(False)
        vgg_extractor_style=accelerator.prepare(vgg_extractor_style)
        
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

        # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
        num_inference_steps = args.num_inference_steps
        #images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images
        #images[0].save("image.png")
        data=load_dataset(args.style_dataset,split="train")
        content_data=load_dataset(args.content_dataset,split="train")
        STYLE_LORA="style_lora"
        CONTENT_LORA="content_lora"
        style_score_list=[]
        style_mse_list=[]
        content_score_list=[]
        content_mse_list=[]
        clip_list=[]
        for k,content_row in enumerate(content_data):
            for i, row in enumerate(data):
                content_image=content_row["image_0"].convert("RGB")
                try:
                    vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
                    vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16').to(accelerator.device)
                except:
                
                    vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16',force_download=True)
                    vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16',force_download=True).to(accelerator.device)
                vit_model.eval()
                vit_model.requires_grad_(False)

                vit_model=accelerator.prepare(vit_model)

                

                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

                mtcnn = BetterMTCNN(device=accelerator.device).to(dtype=torch_dtype)
                mtcnn.eval()
                resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dtype=torch_dtype,device=accelerator.device)
                resnet.eval()

                mtcnn,resnet=accelerator.prepare(mtcnn,resnet)



                if args.pretrained_type=="consistency":
                    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
                elif args.pretrained_type=="stable":
                    try:
                        pipe=StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
                    except:
                        pipe=StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5",force_download=True)
                # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
                pipe.to(torch_device="cuda", torch_dtype=torch_dtype)
                pipe.run_safety_checker=run_safety_checker
                hooks = []
                content_label=content_row["label"]
            
                accelerator.free_memory()
                if i<args.start or i>=args.limit:
                    continue
                label=row["label"]+content_label
                n_image=4
                try:
                    images=[row[f"image_{k}"] for k in range(n_image)]
                except:
                    n_image=1
                    images=[row[f"image_{k}"] for k in range(n_image)]

                images=[image.convert("RGB") for image in images]

                _,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images+[content_image],False)
                vit_style_embedding_list=vit_style_embedding_list[:-1]
                style_embedding=torch.stack(vit_style_embedding_list).mean(dim=0)
                vgg_style_embedding=torch.stack([get_vgg_embedding(vgg_extractor_style,image).clone().detach() for image in images]).mean(dim=0)

                content_embedding=vit_content_embedding_list[-1]
                print("content embedding shape ",content_embedding.size())
                evaluation_images=[]

                mtcnn_image_transforms = transforms.Compose(
                        [
                            transforms.Resize(160, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(160),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ]
                    )
                
                
                
                content_image_tensor=mtcnn_image_transforms(content_image).to(dtype=torch_dtype,device=accelerator.device)

                content_face_embedding=get_face_embeddings(content_image_tensor,resnet,mtcnn,False).detach()

                if args.content_reward_fn=="dino":
                    dino_vit_extractor=ViTExtractor("vit_base_patch16_224",device=accelerator.device)
                    dino_vit_extractor.model.eval()
                    dino_vit_extractor.model.requires_grad_(False)
                    dino_vit_prepocessed=dino_vit_extractor.preprocess_pil(content_image.resize((args.image_size,args.image_size))).to(dtype=torch_dtype,device=accelerator.device)
                    dino_vit_features=dino_vit_extractor.extract_descriptors(dino_vit_prepocessed,facet=args.facet)
                    print("dino vit features",dino_vit_features.size())
                    
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
                sd_pipeline.run_safety_checker=run_safety_checker
                print("before ",sd_pipeline.unet.config.sample_size)
                sd_pipeline.unet.config.sample_size=args.image_size // sd_pipeline.vae_scale_factor
                print("after", sd_pipeline.unet.config.sample_size)
                sd_pipeline.unet.to(accelerator.device).requires_grad_(False)
                sd_pipeline.text_encoder.to(accelerator.device).requires_grad_(False)
                sd_pipeline.vae.to(accelerator.device).requires_grad_(False)

                vae_image_transforms = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),  # Resize to 768x768
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                    ])
                


                if args.prompt_embedding_conditioning or args.use_encoder_hid_proj:
                    
                    if args.image_embeds_type=="face":
                        image_embed_dim=512
                        _src_embedding=content_face_embedding
                    elif args.image_embeds_type=="vit":
                        image_embed_dim=151296
                        _src_embedding=content_embedding
                    
                if args.prompt_embedding_conditioning:
                    prompt_model=PromptImageProjection(image_embed_dim,768,args.num_image_text_embeds)
                    prompt_model.to(accelerator.device).requires_grad_(True)
                    prompt_model=accelerator.prepare(prompt_model)
                    sd_pipeline.register_prompt_model(prompt_model,_src_embedding)
                    print(f"prompt model has {len([p for p in prompt_model.parameters()])} traanable parameters")

                elif args.use_encoder_hid_proj:
                    encoder_hid_proj=ImageProjection(image_embed_dim,768,args.num_image_text_embeds)
                    encoder_hid_proj.to(accelerator.device).requires_grad_(True)
                    encoder_hid_proj=accelerator.prepare(encoder_hid_proj)
                    sd_pipeline.register_encoder_hid_proj(encoder_hid_proj,_src_embedding)
                    print(f"encoder model has {len([p for p in encoder_hid_proj.parameters()])} traanable parameters")

                sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae=accelerator.prepare(sd_pipeline.unet,sd_pipeline.text_encoder,sd_pipeline.vae)
                
                #print('sd_pipeline.unet.config.in_channels',sd_pipeline.unet.config.in_channels)
                #print('sd_pipeline.unet.config.out_channels',sd_pipeline.unet.config.out_channels)

                vae_content_embedding=sd_pipeline.vae.encode(vae_image_transforms(content_image).unsqueeze(0).to(device=accelerator.device, dtype=torch_dtype))

                if args.content_reward_fn=="dift":
                    dift_featurizer=SDFeaturizer()
                    dift_featurizer.pipe.unet.to(device=accelerator.device,dtype=torch_dtype)
                    dift_featurizer.pipe.vae.to(device=accelerator.device,dtype=torch_dtype)
                    dift_featurizer.pipe.text_encoder.to(device=accelerator.device,dtype=torch_dtype)
                    raw_content=vae_image_transforms(content_image).to(device=accelerator.device, dtype=torch_dtype)

                    sd_dift_content= dift_featurizer.forward(raw_content.clone().to(device=accelerator.device, dtype=torch_dtype),
                        prompt="portrait",
                        t=args.t,
                        up_ft_index=args.up_ft_index,
                        ensemble_size=args.ensemble_size) 
                    print("sd_dift_content",sd_dift_content.size())
                    sd_dift_content.requires_grad_(True)


                content_cache=[]
                style_cache=[]
                accelerator.free_memory()
                if args.style_layers_train:

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
                            sample_embedding_list=[get_vgg_embedding(vgg_extractor_style,image) for image in images]
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
                            sample_embedding_list=[get_vgg_embedding(vgg_extractor_style,image) for image in images]
                            
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
                        
                        style_trainer=BetterAlignPropTrainer(
                            align_config,
                            style_reward_function_align,
                            prompt_fn,
                            style_ddpo_pipeline,
                            get_image_logger_align(STYLE_LORA+label,accelerator,style_cache)
                            )
                    
                if args.content_layers_train:


                    @torch.no_grad()
                    def content_reward_function(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any)->  tuple[list[torch.Tensor],Any]:
                        _,__,sample_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
                        return [cos_sim_rescaled(sample,content_embedding) for sample in sample_vit_content_embedding_list],{}

                    def content_reward_function_align(images:torch.Tensor, prompts:tuple[str], metadata:tuple[Any],prompt_metadata:Any=None)->tuple[torch.Tensor,Any]:
                        if args.content_reward_fn=="face":
                            face_embedding_list=[get_face_embeddings(image,resnet,mtcnn) for image in images]
                            return torch.stack([mse_reward_fn(face_embedding,content_face_embedding) for face_embedding in face_embedding_list]),{}
                        elif args.content_reward_fn=="vae":
                            #print("vae_content_embedding.latent_dist.sample()",vae_content_embedding.latent_dist.sample().size())
                            #print("image.unsqueeze(0)",images[0].unsqueeze(0).size())
                            #print(type(images[0].unsqueeze(0)))
                            return torch.stack([mse_reward_fn(vae_content_embedding.latent_dist.sample(), image.unsqueeze(0)) for image in images]),{}
                        elif args.content_reward_fn=="dino":
                            return torch.stack([mse_reward_fn(dino_vit_features,dino_vit_extractor.extract_descriptors(image.unsqueeze(0),facet=args.facet)) for image in images]),{}
                        elif args.content_reward_fn=="dift":
                            return torch.stack([mse_reward_fn(sd_dift_content,  dift_featurizer.forward( 
                                image.unsqueeze(0),prompt="portrait",
                                t=args.t,
                                up_ft_index=args.up_ft_index,
                                ensemble_size=args.ensemble_size)) for image in images]),{}
                        elif args.content_reward_fn=="raw":
                            return torch.stack([mse_reward_fn(raw_content,image) for image in images]),{}
                        #if args.reward_fn=="cos" or args.reward_fn=="mse":
                        _,__,sample_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
                        if args.content_reward_fn=="mse":
                            reward_fn=mse_reward_fn
                        elif args.content_reward_fn=="cos":
                            reward_fn=cos_sim_rescaled
                        return torch.stack([reward_fn(sample,content_embedding) for sample in sample_vit_content_embedding_list]),{}
                    
                    content_keywords=[CONTENT_LORA]
                    sd_pipeline.unet=apply_lora(sd_pipeline.unet,content_layers,[],args.content_mid_block,keyword=CONTENT_LORA)

                    if args.content_reward_fn=="vae":
                        output_type="latent"
                    else:
                        output_type="pt"
                    content_ddpo_pipeline=KeywordDDPOStableDiffusionPipeline(sd_pipeline,[CONTENT_LORA],output_type=output_type)
                    print("n trainable layers content",len(content_ddpo_pipeline.get_trainable_layers()))
                    sd_pipeline.unet.to(accelerator.device)
                    kwargs={}
                    
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
                            get_image_logger_align(CONTENT_LORA+label,accelerator,content_cache)
                        )
                for model in [sd_pipeline,sd_pipeline.unet, sd_pipeline.vae,sd_pipeline.text_encoder]:
                    model.to(accelerator.device)
                total_start=time.time()
                try:
                    for e in range(args.epochs):
                        accelerator.free_memory()
                        start=time.time()
                        if args.style_layers_train:
                            try:
                                style_trainer.train(**kwargs)
                            except torch.cuda.OutOfMemoryError:
                                print("oom epoch ",e)
                                accelerator.free_memory()
                                style_trainer.train(**kwargs)
                            accelerator.free_memory()
                        if args.content_layers_train:
                            try:
                                content_trainer.train(**kwargs)
                            except torch.cuda.OutOfMemoryError:
                                print("oom epoch ",{e})
                                accelerator.free_memory()
                                content_trainer.train(**kwargs)

                            accelerator.free_memory()
                        end=time.time()
                        print(f"\t {label} epoch {e} elapsed {end-start}")
                except  torch.cuda.OutOfMemoryError:
                    print(f"failed after {e} epochs")
                    end=time.time()
                print(f"all epochs for {label} elapsed {end-total_start}")
                sd_pipeline.unet.requires_grad_(False)
                with torch.no_grad():
                    if args.use_unformatted_prompts:
                        for unformatted_prompt in unformatted_prompt_list:
                            prompt=unformatted_prompt.format(args.prompt)
                            image =sd_pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=args.guidance_scale,height=args.image_size,width=args.image_size).images[0]
                            evaluation_images.append(image)
                    else:
                        for _ in range(args.n_evaluation):

                            image=sd_pipeline(prompt=args.prompt, num_inference_steps=num_inference_steps, guidance_scale=args.guidance_scale,height=args.image_size,width=args.image_size).images[0]
                            evaluation_images.append(image)
                

                for image in evaluation_images:
                    accelerator.log({f"evaluation_{label}":wandb.Image(image)})
                for _content_image in content_cache:
                    accelerator.log({f"cache_{label}_{CONTENT_LORA}":wandb.Image(_content_image)})
                for _style_image in style_cache:
                    accelerator.log({f"cache_{label}_{STYLE_LORA}":wandb.Image(_style_image)})
                _,evaluation_vit_style_embedding_list,evaluation_vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,evaluation_images,False)
                style_score=np.mean([cos_sim_rescaled(sample,style_embedding).cpu() for sample in evaluation_vit_style_embedding_list])
                content_score=np.mean([cos_sim_rescaled(sample, content_embedding).cpu() for sample in evaluation_vit_content_embedding_list])
                style_mse=np.mean([F.mse_loss(sample,style_embedding).cpu() for sample in evaluation_vit_style_embedding_list])
                content_mse=np.mean([F.mse_loss(sample, content_embedding).cpu() for sample in evaluation_vit_content_embedding_list])
                metrics={
                    f"{label}_content":content_score,
                    f"{label}_style":style_score,
                    f"{label}_content_mse":content_mse,
                    f"{label}_style_mse":style_mse
                }
                if args.use_unformatted_prompts:
                    for unformatted_prompt,image in zip(unformatted_prompt_list,evaluation_images):
                        inputs = clip_processor(text=[unformatted_prompt], images=image, return_tensors="pt", padding=True)
                        outputs = clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        clip_alignment=logits_per_image.item()
                        metrics[f"{label}_clip_alignment"]=clip_alignment
                        clip_list.append(clip_alignment)
                accelerator.log(metrics)
                print(metrics)
                content_score_list.append(content_score)
                content_mse_list.append(content_mse)
                style_score_list.append(style_score)
                style_mse_list.append(style_mse)
                sd_pipeline.unet,sd_pipeline= accelerator.free_memory(sd_pipeline.unet,sd_pipeline)
                if args.style_layers_train:
                    style_trainer,style_ddpo_pipeline=accelerator.free_memory(style_trainer,style_ddpo_pipeline)
                if args.content_layers_train:
                    content_trainer,content_ddpo_pipeline=accelerator.free_memory(content_trainer,content_ddpo_pipeline)
        metrics={
            f"content":np.mean(content_score_list),
            f"style":np.mean(style_score_list),
            f"content_mse":np.mean(content_mse_list),
            f"style_mse":np.mean(style_mse_list)
            }
        if args.use_unformatted_prompts:
            metrics[f"clip_alignment"]=np.mean(clip_list)
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