import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from pipelines import CompatibleLatentConsistencyModelPipeline
from datasets import load_dataset
import torchvision.transforms as transforms
import torch
from accelerate import Accelerator
import time
from diffusers.models.embeddings import IPAdapterFullImageProjection
from extractor import ViTExtractor
import torch.nn.functional as F
from PIL import Image
import random

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--epochs",type=int,default=1)
parser.add_argument("--embedding",type=str,default="dino")
parser.add_argument("--reward_embedding",type=str,default="dino")
parser.add_argument("--facet",type=str,default="token",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--data_dir",type=str,default="data_dir")
parser.add_argument("--save_data_npz",action="store_true")
parser.add_argument("--load_data_npz",action="store_true")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--pipeline",type=str,default="lcm")

import torch
import torch.nn.functional as F

def make_batches(tensor_list, batch_size, pad_value=0):
    """
    Splits a list of tensors into batches, padding them to the same shape within each batch.

    Args:
        tensor_list (List[torch.Tensor]): List of tensors to batch. Each tensor can have different shapes.
        batch_size (int): Desired batch size.
        pad_value (int, float): Value to use for padding shorter tensors.

    Returns:
        List[torch.Tensor]: List of batched tensors, each of shape (batch_size, *max_shape).
    """
    batches = []
    for i in range(0, len(tensor_list), batch_size):
        batch = tensor_list[i:i+batch_size]
        # Determine max shape in this batch
        max_shape = list(batch[0].shape)
        for tensor in batch[1:]:
            for dim, size in enumerate(tensor.shape):
                max_shape[dim] = max(max_shape[dim], size)

        # Create padded batch
        padded = []
        for tensor in batch:
            # Compute pad widths for F.pad, in reverse order: (pad_last_dim_left, pad_last_dim_right, pad_second_last_left, ...)
            pad_sizes = []
            for size, max_size in zip(tensor.shape[::-1], max_shape[::-1]):
                pad_sizes.extend([0, max_size - size])
            padded_tensor = F.pad(tensor, pad_sizes, value=pad_value)
            padded.append(padded_tensor)

        # Stack into a single tensor
        batched = torch.stack(padded, dim=0)
        batches.append(batched)

    return batches


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    raw_data=load_dataset(args.dataset,split="train")

    os.makedirs(args.data_dir,exist_ok=True)

    if args.embedding=="dino":
        dino_vit_extractor=ViTExtractor("vit_base_patch16_224",device=accelerator.device)
        dino_vit_extractor.model.eval()
        dino_vit_extractor.model.requires_grad_(False)

    def embed_img_tensor(img_tensor:torch.Tensor)->torch.Tensor:
        if args.embedding=="dino":
            if len(img_tensor.size())==3:
                img_tensor=img_tensor.unsqueeze(0)
            img_tensor=F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            #dino_vit_prepocessed=dino_vit_extractor.preprocess_pil(content_image.resize((args.image_size,args.image_size))).to(dtype=torch_dtype,device=accelerator.device)
            dino_vit_features=dino_vit_extractor.extract_descriptors(img_tensor,facet=args.facet)
            batch_size=img_tensor.size()[0]
            embedding=dino_vit_features.view(batch_size,-1)
        return embedding
    
    def transform_image(pil_image:Image.Image):
        if args.embedding=="dino":
            t=transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize(dino_vit_extractor.mean,dino_vit_extractor.std)]
            )
        return t(pil_image)
    
    embedding_list=[]
    text_list=[]
    shuffled_row_list=[row for row in raw_data]
    random.shuffle(shuffled_row_list)
    for row in raw_data:
        image=row["image"]
        text=row["text"]
        embedding_list.append(embed_img_tensor(transform_image(image)))
        text_list.append(text)

    def loss_fn(img_tensor_batch:torch.Tensor, src_embedding_batch:torch.Tensor)->torch.Tensor:
        pred_embedding_batch=embed_img_tensor(img_tensor_batch)
        return F.mse_loss(pred_embedding_batch,src_embedding_batch)
    
    fake_image=torch.rand((1,3,args.image_size,args.image_size))
    fake_embedding=embed_img_tensor(fake_image)
    embedding_dim=fake_embedding.size()[-1]

    print("embedding dim",embedding_dim)

    projection_layer=IPAdapterFullImageProjection(embedding_dim)

    #the output of the embeddign thing can be passed as ip_adapter_image_embeds or the image itself can be passed as     ip_adapter_image to the pipeline
    #multiple projection layers for different layers..?

    if args.pipeline=="lcm":
        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        #todo: compatible SanaSprint

    


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