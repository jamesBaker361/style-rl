#given 128 x 128, resize to 256 x 256 so its blurry
#light weight post-processing module

import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from adapter_helpers import replace_ip_attn,get_modules_of_types
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from gpu_helpers import *
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC,UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from peft import LoraConfig
import torch.nn.functional as F

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from embedding_helpers import EmbeddingUtil
from data_helpers import ScaleDataset
from torchviz import make_dot
torch.autograd.set_detect_anomaly(True)

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--project_name",type=str,default="refiner")
parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--dataset",type=str,default="jlbaker361/siglip2-art_coco_captioned")
parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
parser.add_argument("--intermediate_embedding_dim",type=int,default=1024)
parser.add_argument("--cross_attention_dim",type=int,default=768)
parser.add_argument("--num_image_text_embeds",type=int,default=4)
parser.add_argument("--disable_projection_adapter",action="store_true",help="whether to use projection for ip adapter ")
parser.add_argument("--identity_adapter",action="store_true",help="whether to use identity mapping for IP adapter layers")
parser.add_argument("--deep_to_ip_layers",action="store_true",help="use deeper ip layers")
parser.add_argument("--train_split",type=float,default=0.5)
parser.add_argument("--epochs",type=int,default=4)
parser.add_argument("--use_lora",action="store_true")
parser.add_argument("--patch_scale",type=int,default=8)
parser.add_argument("--validation_interval",type=int,default=2)
parser.add_argument("--num_inference_steps",type=int,default=2)
parser.add_argument("--limit",type=int,default=8)
parser.add_argument("--embedding",type=str,default="siglip2",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--verbose",action="store_true")


def image_to_patches(img, patch_size):
    """
    img: Tensor of shape (C, H, W)
    patch_size: int, size of each patch (assumes square patches)
    Returns: Tensor of shape (N_patches, C, patch_size, patch_size)
    """
    C, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image must be divisible by patch size"

    # (C, H, W) → (1, C, H, W)
    img = img.unsqueeze(0)

    # Apply unfold: (1, C, num_patches_H, num_patches_W, patch_H, patch_W)
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)  # (C, N, P, P)
    patches = patches.permute(1, 0, 2, 3)  # (N, C, P, P)
    
    return patches

def patches_to_image(patches, image_height, image_width):
    """
    Reconstructs an image from non-overlapping patches.

    Args:
        patches: Tensor of shape (N, C, P, P)
        image_height: Original image height (e.g., 256)
        image_width: Original image width (e.g., 256)

    Returns:
        Tensor of shape (C, H, W)
    """
    N, C, P, _ = patches.shape
    patch_rows = image_height // P
    patch_cols = image_width // P

    # (N, C, P, P) → (patch_rows, patch_cols, C, P, P)
    patches = patches.view(patch_rows, patch_cols, C, P, P)

    # → (C, patch_rows, P, patch_cols, P)
    patches = patches.permute(2, 0, 3, 1, 4)

    # → (C, H, W)
    image = patches.reshape(C, image_height, image_width)

    return image

def split_list_by_ratio(lst, ratios=(0.8, 0.1, 0.1)):
    #assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    n = len(lst)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return lst[:i1], lst[i1:i2], lst[i2:]

def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    try:
        raw_data=load_dataset(args.dataset,split="train")
    except OSError:
        raw_data=load_dataset(args.dataset,split="train",download_mode="force_redownload")
    WEIGHTS_NAME="unet_model.bin"
    CONFIG_NAME="config.json"
    save_dir=os.path.join(os.environ["TORCH_LOCAL_DIR"],args.name)
    save_path=os.path.join(save_dir,WEIGHTS_NAME)
    config_path=os.path.join(save_dir,CONFIG_NAME)

    #pipeline.requires_grad_(False)
    embedding_list=[]
    image_list=[]
    posterior_list=[]
    shuffled_row_list=[row for row in raw_data]

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

    with torch.no_grad():
        for i,row in enumerate(shuffled_row_list):
            if i==args.limit:
                break
            before_objects=find_cuda_objects()
            image=row["image"]
            
            
            
            if "embedding" in row:
                #print(row["embedding"])
                np_embedding=np.array(row["embedding"])[-1]
                #print("np_embedding",np_embedding.shape)
                embedding=torch.from_numpy(np_embedding)
                #print("embedding",embedding.size())
                #real_embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0)
                #print("real embedding",real_embedding.size())
            else:
                #this should NOT be normalized or transformed
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image))[-1]
            image=pipeline.image_processor.preprocess(image).squeeze(0)
            image_list.append( image)
            #print(embedding.size())
            embedding=embedding.to("cpu") #.squeeze()
            embedding_list.append(embedding)
            accelerator.free_memory()
            torch.cuda.empty_cache()


            if i ==1:
                accelerator.print("embedding size",embedding.size(),"img size",image.size())

            #print(get_gpu_memory_usage())
            #print("gpu objects:",len(find_cuda_objects()))
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)

    pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    unet=pipeline.unet.to(device)
    scheduler=pipeline.scheduler
    if args.use_lora:
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],)

        unet.add_adapter(unet_lora_config)
    unet.conv_in=torch.nn.Conv2d(3,unet.conv_in.out_channels,
                                 unet.conv_in.kernel_size,
                                 unet.conv_in.stride,
                                 unet.conv_in.padding)
    unet.conv_in.requires_grad_(True)

    unet.conv_out=torch.nn.Conv2d(unet.conv_out.in_channels,3,
                                  unet.conv_out.kernel_size,
                                  unet.conv_out.stride,
                                  unet.conv_out.padding)
    unet.conv_out.requires_grad_(True)
    params=list(set([p for p in unet.parameters() if p.requires_grad]))
    accelerator.print("len params",len(params))
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin",low_cpu_mem_usage=False,ignore_mismatched_sizes=True)

    params=list(set([p for p in unet.parameters() if p.requires_grad]+[p for p in unet.encoder_hid_proj.parameters() if p.requires_grad]))
    accelerator.print("len params",len(params))

    fake_image=torch.rand((1,3,args.image_size,args.image_size))
    fake_embedding=embedding_util.embed_img_tensor(fake_image)
    embedding_dim=fake_embedding.size()[-1]

    if args.disable_projection_adapter:
        use_projection=False
    else:
        use_projection=True

    cross_attention_dim=args.cross_attention_dim
    if args.identity_adapter:
        cross_attention_dim=embedding_dim//args.num_image_text_embeds
    intermediate_embedding_dim=args.intermediate_embedding_dim
    
    accelerator.print("embedding_dim",embedding_dim)
    accelerator.print("cross_attention_dim",cross_attention_dim)
    accelerator.print("intermediate",intermediate_embedding_dim)

    if use_projection and args.identity_adapter:
        accelerator.print("use_projection and args.identity_adapter are both true")

    replace_ip_attn(unet,
                    embedding_dim,
                    intermediate_embedding_dim,
                    cross_attention_dim,
                    args.num_image_text_embeds,
                    use_projection,args.identity_adapter,args.deep_to_ip_layers)

    params=list(set([p for p in unet.parameters() if p.requires_grad]+[p for p in unet.encoder_hid_proj.parameters() if p.requires_grad]))
    accelerator.print("len params",len(params))


    

    ratios=(args.train_split,(1.0-args.train_split)/2.0,(1.0-args.train_split)/2.0)
    accelerator.print('train/test/val',ratios)
    #batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
    embedding_list,test_embedding_list,val_embedding_list=split_list_by_ratio(embedding_list,ratios)
    
    image_list,test_image_list,val_image_list=split_list_by_ratio(image_list,ratios)

    patch_size=args.image_size//args.patch_scale
    print(f"args.image_size {args.image_size} // args.patch_scale {args.patch_scale} ={patch_size}")

    def patchify_lists(old_image_list,old_embedding_list):
        new_image_list=[]
        new_embedding_list=[]
        for old_image,old_embedding in zip(old_image_list,old_embedding_list):
            patches=image_to_patches(old_image,patch_size)
            for p in patches:
                new_image_list.append(p)
                new_embedding_list.append(old_embedding.clone())
        return new_image_list,new_embedding_list

    image_list,embedding_list=patchify_lists(image_list,embedding_list)
    val_image_list,val_embedding_list=patchify_lists(val_image_list,val_embedding_list)
    test_image_list,test_embedding_list=patchify_lists(test_image_list,test_embedding_list)

    pipeline.text_encoder.to(device)
    pipeline.text_encoder.requires_grad_(False)

    unconditioned_text_embeds,negative_text_embeds=pipeline.encode_prompt(
                                       prompt= " ",
                                        device=device, #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= True,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
    unconditioned_text_embeds=unconditioned_text_embeds.squeeze(0).cpu().detach()
    accelerator.print("text embeds",unconditioned_text_embeds.size())
    
    train_dataset=ScaleDataset(image_list=image_list,embedding_list=embedding_list,text_embedding=unconditioned_text_embeds)
    val_dataset=ScaleDataset(image_list=val_image_list,embedding_list=val_embedding_list,text_embedding=unconditioned_text_embeds)
    test_dataset=ScaleDataset(image_list=test_image_list,embedding_list=test_embedding_list,text_embedding=unconditioned_text_embeds)

    for dataset_batch in train_dataset:
        break

    accelerator.print("dataset batch",type(dataset_batch))

    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=patch_size**2)
    test_loader=DataLoader(test_dataset,batch_size=patch_size**2) #each test batch is one image

    for train_batch in train_loader:
        break
    for val_batch in val_loader:
        break

    accelerator.print("train batch",type(train_batch),train_batch["image"].size())
    accelerator.print("val batch",type(val_batch),val_batch["image"].size())

    optimizer=torch.optim.AdamW(params,lr=args.lr)
    unet=unet.to(device)

    train_loader,test_loader,val_loader,optimizer,unet=accelerator.prepare(train_loader,test_loader,val_loader,optimizer,unet)

    @torch.no_grad()
    def forward(unet,noise:torch.Tensor,embedding,scheduler,num_inference_steps:int)->torch.Tensor:
        added_cond_kwargs={"image_embeds":embedding}
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            noise_pred=unet(noise,t,added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]
            noise=scheduler.step(noise_pred, t, noise,  return_dict=False)[0]

        return noise

    @torch.no_grad()
    def logging(loader,unet,scheduler):
        pil_image_list=[]
        logging_loss_buffer=[]
        for b,batch in enumerate(loader):
            if b==args.limit:
                break
            image_patches=batch["image"]
            embeddings=batch["embedding"]

            if random.random()<0.5:
                down_scale_factor=0.5
            else:
                down_scale_factor=0.25

            up_scale_factor=1.0/down_scale_factor

            # For shape (B, C, H, W)
            lowres = F.interpolate(image_patches, scale_factor=down_scale_factor, mode='bilinear', align_corners=False)
            upscaled = F.interpolate(lowres, scale_factor=up_scale_factor, mode='bilinear', align_corners=False)

            mini_batches=[
                forward(unet,upscaled[i:i+args.batch_size],embeddings,scheduler,args.num_inference_steps )for i in range(0,(patch_size**2)//args.batch_size,args.batch_size)
            ]
            processed_patches=torch.cat(mini_batches)


            logging_loss_buffer.append(F.mse_loss(processed_patches.float(), image_patches.float()).cpu().detach().numpy())
            processed_image=patches_to_image(processed_patches,args.image_size,args.image_size)
            pil_image=pipeline.image_processor.postprocess(processed_image)
            pil_image_list.append(pil_image)
        return pil_image_list,logging_loss_buffer
    
    
    #accelerator.print(unet)
    start_epoch=1
    for e in range(start_epoch, args.epochs+1):
        loss_buffer=[]
        for b,batch in enumerate(train_loader):
            #with accelerator.accumulate(params):
            if b==args.limit:
                break
            embedding_batch=batch["embedding"].to(device)
            images_batch=batch["image"].to(device)
            encoder_hidden_states=batch["text_embedding"].to(device)
            bsz=images_batch.size()[0]

            if e==1 and b==0:
                accelerator.print('images.size()',images_batch.size())
                accelerator.print('embedding.size()',embedding_batch.size())


            if random.random()<0.5:
                down_scale_factor=0.5
            else:
                down_scale_factor=0.25

            up_scale_factor=1.0/down_scale_factor

            with torch.no_grad():
                lowres = F.interpolate(images_batch.clone().detach(), scale_factor=down_scale_factor, mode='bilinear', align_corners=False)
                upscaled = F.interpolate(lowres, scale_factor=up_scale_factor, mode='bilinear', align_corners=False).detach()

                #upscaled=torch.randn_like(images_batch)

                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=images_batch.device)
                timesteps = timesteps.long()

                noisy_images=scheduler.add_noise(images_batch, upscaled,timesteps)

                added_cond_kwargs={"image_embeds":embedding_batch}

                if scheduler.config.prediction_type == "epsilon":
                    target = upscaled.detach()
                elif scheduler.config.prediction_type == "v_prediction":
                    target = scheduler.get_velocity(images_batch, upscaled, timesteps).detach()
            
            target=target.detach()
            model_pred = unet(noisy_images, timesteps, 
                            added_cond_kwargs=added_cond_kwargs, 
                            encoder_hidden_states=encoder_hidden_states,
                            return_dict=False)[0]

            

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if args.verbose:

                print("=== Debug Info ===")
                print(f"Batch {b}, Epoch {e}")
                print(f"Loss requires_grad: {loss.requires_grad}")
                print(f"Loss is_leaf: {loss.is_leaf}")
                print(f"Loss grad_fn: {loss.grad_fn}")

                # Check if any tensors are being reused
                print(f"Target requires_grad: {target.requires_grad}")
                print(f"Model_pred requires_grad: {model_pred.requires_grad}")
                print("model grad_fn",model_pred.grad_fn)

                print("=== Parameter Check ===")
                corrupted_params = []
                for name, param in unet.named_parameters():
                    if param.grad_fn is not None or not param.is_leaf:
                        corrupted_params.append(name)
                        print(f"CORRUPTED: {name} - grad_fn: {param.grad_fn}, is_leaf: {param.is_leaf}")

                if corrupted_params:
                    print(f"Found {len(corrupted_params)} corrupted parameters!")

            accelerator.backward(loss)
            # Only step optimizer after accumulation steps
            #if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()

            loss_buffer.append(loss.cpu().detach().item())
        end=time.time()
        elapsed=end-start
        accelerator.print(f"\t epoch {e} elapsed {elapsed}")
        accelerator.log({
            "loss_mean":np.mean(loss_buffer),
            "loss_std":np.std(loss_buffer),
        })

        torch.cuda.empty_cache()
        accelerator.free_memory()
        if e%args.validation_interval==0:
            val_image_list,val_loss_buffer=logging(val_loader,unet,scheduler)
            accelerator.log({
                "val_loss_mean":np.mean(val_loss_buffer),
                "val_loss_std":np.std(val_loss_buffer),
            })
            for k,val_image in enumerate(val_image_list):
                accelerator.log({
                    f"val_{k}":wandb.Image(val_image)
                })

        test_image_list,test_loss_buffer=logging(test_loader,unet,scheduler)
        accelerator.log({
            "test_loss_mean":np.mean(test_loss_buffer),
            "test_loss_std":np.std(test_loss_buffer)
        })
        for k,test_image in enumerate(test_image_list):
            accelerator.log({
                f"test_{k}":wandb.Image(test_image)
            })
        accelerator.print({
            "test_loss_mean":np.mean(test_loss_buffer),
            "test_loss_std":np.std(test_loss_buffer)
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