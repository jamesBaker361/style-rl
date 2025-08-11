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

def concat_images_horizontally(images):
    """
    Concatenate a list of PIL.Image objects horizontally.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images concatenated side-by-side.
    """
    # Resize all images to the same height (optional)
    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [
        img if img.height == min_height else img.resize(
            (int(img.width * min_height / img.height), min_height),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total width and max height
    total_width = sum(img.width for img in resized_images)
    height = min_height

    # Create new blank image
    new_img = Image.new('RGB', (total_width, height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img

def concat_images_vertically(images):
    """
    Concatenate a list of PIL.Image objects vertically.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images stacked top-to-bottom.
    """
    # Resize all images to the same width (optional)
    widths = [img.width for img in images]
    min_width = min(widths)
    resized_images = [
        img if img.width == min_width else img.resize(
            (min_width, int(img.height * min_width / img.width)),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total height and max width
    total_height = sum(img.height for img in resized_images)
    width = min_width

    # Create new blank image
    new_img = Image.new('RGB', (width, total_height))

    # Paste images one below the other
    y_offset = 0
    for img in resized_images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.height

    return new_img


name="jlbaker361/denoise_epsilon_ssl_1.0_0.001_1000_identity_lcm_-1"

accelerator=Accelerator(mixed_precision="fp16")
with accelerator.autocast():
    with torch.no_grad():

        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)

        vae=pipeline.vae
        denoising_model=pipeline.unet
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        accelerator.print(pipeline.scheduler)
        pipeline.unet.encoder_hid_proj=None
        text_encoder=pipeline.text_encoder
        scheduler=pipeline.scheduler

        denoising_model.requires_grad_(False)
        use_projection=True

        embedding_dim=1536

        cross_attention_dim=768
        cross_attention_dim=embedding_dim//4
        intermediate_embedding_dim=1024
        intermediate_embedding_dim=embedding_dim

        WEIGHTS_NAME="unet_model.bin"
        CONFIG_NAME="config.json"
        save_dir=os.path.join(os.environ["TORCH_LOCAL_DIR"],name)
        save_path=os.path.join(save_dir,WEIGHTS_NAME)

        replace_ip_attn(denoising_model,
                        embedding_dim,
                        intermediate_embedding_dim,
                        cross_attention_dim,
                        4,
                        use_projection,True,False)
        #print("image projection",unet.encoder_hid_proj.multi_ip_adapter.image_projection_layers[0])
        start_epoch=1
        persistent_loss_list=[]
        persistent_grad_norm_list=[]
        persistent_text_alignment_list=[]
        persistent_fid_list=[]
        denoising_model.load_state_dict(torch.load(save_path,weights_only=True),strict=False)

        raw_data=load_dataset("jlbaker361/ssl-league_captioned_splash-1000",split="train")

        embedding_list=[]
        text_list=[]
        image_list=[]
        posterior_list=[]
        prompt_list=[]
        shuffled_row_list=[row for row in raw_data]

        embedding_util=EmbeddingUtil(accelerator.device,torch.float16,"ssl",None,None)

        for i,row in enumerate(shuffled_row_list):
            if i==10:
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

            image=pipeline.image_processor.preprocess(image)[0]
            if "posterior" not in row:
                posterior=public_encode(vae,image).squeeze(0)
            else:
                np_posterior=np.array(row["posterior"])
                posterior=torch.from_numpy(np_posterior)
            posterior=posterior.to("cpu")
            posterior_list.append(posterior)
            image_list.append(image.squeeze(0))
            #print(embedding.size())
            embedding=embedding.to("cpu") #.squeeze()
            embedding_list.append(embedding)
            accelerator.free_memory()
            torch.cuda.empty_cache()

            text=row["text"]
            if type(text)==str:
                prompt=text
                text, _ = pipeline.encode_prompt(
                                        text,
                                        "cpu", #accelerator.device,
                                        1,
                                        pipeline.do_classifier_free_guidance,
                                        negative_prompt=None,
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
            else:
                np_text=np.array(text)
                text=torch.from_numpy(np_text)
                prompt=row["prompt"]
            text=text.to("cpu").squeeze(0)
            if i ==1:
                accelerator.print("text size",text.size(),"embedding size",embedding.size(),"img size",image.size(),"latent size",posterior.size())
            #print(get_gpu_memory_usage())
            #print("gpu objects:",len(find_cuda_objects()))
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)
            #print("grads",len(find_cuda_tensors_with_grads()))
            
        pipeline.to(torch.float16)
        embedding=embedding.to(torch.float16)
        output=pipeline("going for a walk",256,256,num_inference_steps=10, ip_adapter_image_embeds=[embedding.unsqueeze(0)])
        stacked=torch.stack(output.denoised_list)
        accelerator.print("stacked size",stacked.size())
        pil_image_list=pipeline.image_processor.postprocess(stacked)

        concat1=concat_images_horizontally(pil_image_list)

        #concat.save("denoised_ip.png")

        output=pipeline("going for a walk",256,256,num_inference_steps=10, ip_adapter_image_embeds=None)
        stacked=torch.stack(output.denoised_list)
        accelerator.print("stacked size",stacked.size())
        pil_image_list=pipeline.image_processor.postprocess(stacked)

        concat2=concat_images_horizontally(pil_image_list)

        concat=concat_images_vertically(concat1,concat2)
