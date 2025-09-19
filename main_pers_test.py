import os
import argparse
from experiment_helpers.gpu_details import print_details
from pipelines import CompatibleLatentConsistencyModelPipeline,CompatibleStableDiffusionPipeline
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
from worse_peft import apply_lora
import wandb
import numpy as np
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil
from data_helpers import CustomDataset
from custom_vae import public_encode
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from PIL import Image
from sana_pipelines import CompatibleSanaSprintPipeline, prepare_ip_adapter,compatible_forward_sana_transformer_model
from custom_scheduler import *

from image_utils import concat_images_horizontally


seed=1234
random.seed(seed)                      # Python
np.random.seed(seed)                   # NumPy
torch.manual_seed(seed)                # PyTorch (CPU)
try:
    torch.cuda.manual_seed(seed)           # PyTorch (GPU)
except:
    pass



parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--training_type",help="denoise or reward",default="denoise",type=str)
parser.add_argument("--prediction_type",type=str,default="epsilon",help="epsilon or v_prediction")
parser.add_argument("--train_split",type=float,default=0.5)
parser.add_argument("--validation_interval",type=int,default=40)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--intermediate_embedding_dim",type=int,default=1024)
parser.add_argument("--cross_attention_dim",type=int,default=768)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--num_image_text_embeds",type=int,default=4)
parser.add_argument("--deepspeed",action="store_true",help="whether to use deepspeed")
parser.add_argument("--fsdp",action="store_true",help=" whether to use fsdp training")
parser.add_argument("--vanilla",action="store_true",help="no distribution")
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--load",action="store_true",help="whether to load saved version")
parser.add_argument("--load_hf",action="store_true",help="whether to load saved version from hf")
parser.add_argument("--upload_interval",type=int,default=50,help="how often to upload during training")
parser.add_argument("--generic_test_prompts",action="store_true")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--disable_projection_adapter",action="store_true",help="whether to use projection for ip adapter ")
parser.add_argument("--identity_adapter",action="store_true",help="whether to use identity mapping for IP adapter layers")
parser.add_argument("--deep_to_ip_layers",action="store_true",help="use deeper ip layers")
parser.add_argument("--scheduler_type",type=str,default="LCMScheduler")
parser.add_argument("--reward_switch_epoch",type=int,default=-1)
parser.add_argument("--initial_scale",type=float,default=1.0)
parser.add_argument("--final_scale",type=float,default=1.0)
parser.add_argument("--sigma_data",type=float,default=-0.8)

import torch
import torch.nn.functional as F

def split_list_by_ratio(lst, ratios=(0.8, 0.1, 0.1)):
    #assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    n = len(lst)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return lst[:i1], lst[i1:i2], lst[i2:]




def main(args):
    if args.deepspeed:
        accelerator=Accelerator(log_with="wandb")
        print("using deepspeed")
    else:
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    api=HfApi()


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]



    #with accelerator.autocast():
    try:
        raw_data=load_dataset(args.dataset,split="train")
    except OSError:
        raw_data=load_dataset(args.dataset,split="train",download_mode="force_redownload")
    WEIGHTS_NAME="unet_model.bin"
    CONFIG_NAME="config.json"
    save_dir=os.path.join(os.environ["TORCH_LOCAL_DIR"],args.name)
    save_path=os.path.join(save_dir,WEIGHTS_NAME)
    config_path=os.path.join(save_dir,CONFIG_NAME)
    if accelerator.is_main_process or state.num_processes==1:
        os.makedirs(save_dir,exist_ok=True)

    accelerator.print("\nMODEL-NAME ",args.name.split("/")[-1])
    

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)


    adapter_id = "latent-consistency/lcm-lora-sdv1-5"
    if args.pipeline=="lcm":
        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
    elif args.pipeline=="lcm_post_lora":
        pipeline=CompatibleStableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        pipeline.load_lora_weights(adapter_id)
        pipeline.disable_lora()
    elif args.pipeline=="lcm_pre_lora":
        pipeline=CompatibleStableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        
        pipeline.load_lora_weights(adapter_id)
        pipeline.fuse_lora()
    elif args.pipeline=="sana":
        pipeline = CompatibleSanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        )
        #pipeline.vae=AutoencoderDC.from_pretrained("")
    try:
        pipeline.safety_checker=None
    except Exception as err:
        accelerator.print("tried to set safety checker to None",err)

    if type(pipeline.scheduler)==SCMScheduler:
        pipeline.scheduler=CompatibleSCMScheduler.from_config(pipeline.scheduler.config)
    elif type(pipeline.scheduler)==DEISMultistepScheduler:
        pipeline.scheduler=CompatibleDEISMultistepScheduler.from_config(pipeline.scheduler.config)
    elif type(pipeline.scheduler)==FlowMatchEulerDiscreteScheduler:
        pipeline.scheduler=CompatibleFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)


    accelerator.print(pipeline.scheduler)

    for attribute in ["add_noise","get_velocity","step"]:
        if getattr(pipeline.scheduler,attribute,None) is not None:
            print(f"scheduler {attribute} exists")
        else:
            print(f"scheduler {attribute} does not exist ")
    '''scheduler_class={
            "LCMScheduler":LCMScheduler,
            "DDIMScheduler":DDIMScheduler,
            "DEISMultistepScheduler":DEISMultistepScheduler,
            "CompatibleFlowMatchEulerDiscreteScheduler":CompatibleFlowMatchEulerDiscreteScheduler
    }[args.scheduler_type]
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)
    accelerator.print(pipeline.scheduler)'''

    vae=pipeline.vae
    if args.pipeline=="sana":
        denoising_model=pipeline.transformer
    else:
        denoising_model=pipeline.unet
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        accelerator.print(pipeline.scheduler)
        pipeline.unet.encoder_hid_proj=None
    text_encoder=pipeline.text_encoder
    scheduler=pipeline.scheduler
    

    

    #pipeline.requires_grad_(False)
    embedding_list=[]
    text_list=[]
    image_list=[]
    posterior_list=[]
    prompt_list=[]
    shuffled_row_list=[row for row in raw_data]

    

    do_classifier_free_guidance=False
    if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
        do_classifier_free_guidance=True


    for component in [vae,text_encoder]:
        component.requires_grad_(False)
        component.to("cpu")

    if args.pipeline=="sana":
        unconditioned_text,unconditioned_text_attention_mask=pipeline.encode_prompt(
                                       prompt= " ",
                                        device="cpu", #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= do_classifier_free_guidance,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        negative_text_embeds=None
    else:
        unconditioned_text,negative_text_embeds=pipeline.encode_prompt(
                                       prompt= " ",
                                        device="cpu", #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= do_classifier_free_guidance,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        unconditioned_text_attention_mask=None
    
    
    
    for i in range(len(text_list)):
        if random.random()<=args.uncaptioned_frac:
            text_list[i]=unconditioned_text.squeeze(0).clone().detach()

    def loss_fn(pred_embedding_batch:torch.Tensor, src_embedding_batch:torch.Tensor)->torch.Tensor:
        #pred_embedding_batch=embedding_util.embed_img_tensor(img_tensor_batch)
        return F.mse_loss(pred_embedding_batch.float(),src_embedding_batch.float(),reduce="mean")
    
    fake_image=torch.rand((1,3,args.image_size,args.image_size))
    fake_embedding=embedding_util.embed_img_tensor(fake_image)
    embedding_dim=fake_embedding.size()[-1]

    
    
    denoising_model.requires_grad_(False)
    if args.disable_projection_adapter:
        use_projection=False
    else:
        use_projection=True

    cross_attention_dim=args.cross_attention_dim
    if args.identity_adapter:
        cross_attention_dim=embedding_dim//args.num_image_text_embeds
    intermediate_embedding_dim=args.intermediate_embedding_dim
    if args.disable_projection_adapter:
        intermediate_embedding_dim=embedding_dim

    if use_projection and args.identity_adapter:
        accelerator.print("use_projection and args.identity_adapter are both true")
        

    if args.pipeline=="sana":
        prepare_ip_adapter(pipeline.transformer,accelerator.device,torch_dtype,cross_attention_dim)
    
    replace_ip_attn(denoising_model,
                    embedding_dim,
                    intermediate_embedding_dim,
                    cross_attention_dim,
                    args.num_image_text_embeds,
                    use_projection,args.identity_adapter,args.deep_to_ip_layers)
    #print("image projection",unet.encoder_hid_proj.multi_ip_adapter.image_projection_layers[0])
    start_epoch=1
    persistent_loss_list=[]
    persistent_grad_norm_list=[]
    persistent_text_alignment_list=[]
    persistent_fid_list=[]
    if args.load:
        try:
            denoising_model.load_state_dict(torch.load(save_path,weights_only=True),strict=False)
            with open(config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]+1
            persistent_loss_list=data["persistent_loss_list"]
            persistent_text_alignment_list=data["persistent_text_alignment_list"]
            persistent_fid_list=data["persistent_fid_list"]
            try:
                persistent_grad_norm_list=data["persistent_grad_norm_list"]
            except KeyError:
                print("key error persistent_grad_norm_list list")
            accelerator.print("loaded from ",save_path)
        except Exception as e:
            accelerator.print("couldnt load locally")
            accelerator.print(e)
    if args.load_hf:    
        try:
            pretrained_weights_path=api.hf_hub_download(args.name,WEIGHTS_NAME,force_download=True)
            pretrained_config_path=api.hf_hub_download(args.name,CONFIG_NAME,force_download=True)
            denoising_model.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
            with open(pretrained_config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]+1
            persistent_loss_list=data["persistent_loss_list"]
            persistent_text_alignment_list=data["persistent_text_alignment_list"]
            persistent_fid_list=data["persistent_fid_list"]
            try:
                persistent_grad_norm_list=data["persistent_grad_norm_list"]
            except KeyError:
                print("key error persistent_grad_norm_list list")
            accelerator.print("loaded from  ",pretrained_weights_path)
        except Exception as e:
            accelerator.print("couldnt load from hf")
            accelerator.print(e)

    accelerator.print("start epoch", start_epoch)
        

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