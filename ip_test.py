import os
import argparse
from experiment_helpers.gpu_details import print_details
from pipelines import CompatibleLatentConsistencyModelPipeline
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
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler
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
parser.add_argument("--train_split",type=float,default=0.5)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--deepspeed",action="store_true",help="whether to use deepspeed")
parser.add_argument("--fsdp",action="store_true",help=" whether to use fsdp training")
parser.add_argument("--vanilla",action="store_true",help="no distribution")


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
    #save_path=os.path.join(save_dir,WEIGHTS_NAME)
    #config_path=os.path.join(save_dir,CONFIG_NAME)

    

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    

    adapter_id = "latent-consistency/lcm-lora-sdv1-5"
    pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)

    vae=pipeline.vae
    unet=pipeline.unet
    text_encoder=pipeline.text_encoder
    scheduler=pipeline.scheduler
    accelerator.print(pipeline.scheduler)
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    accelerator.print(pipeline.scheduler)

    #pipeline.requires_grad_(False)
    embedding_list=[]
    text_list=[]
    image_list=[]
    posterior_list=[]
    prompt_list=[]
    shuffled_row_list=[row for row in raw_data]

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
            text_list.append(text)
            prompt_list.append(prompt)
            #print(get_gpu_memory_usage())
            #print("gpu objects:",len(find_cuda_objects()))
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)
            #print("grads",len(find_cuda_tensors_with_grads()))
    
    accelerator.print("prompt list",len(prompt_list))
    accelerator.print("image_list",len(image_list))
    accelerator.print("text_list",len(text_list))
    accelerator.print("posterior list",len(posterior_list))
    accelerator.print("embedding list",len(embedding_list))

    do_classifier_free_guidance=False

    unconditioned_text,_=pipeline.encode_prompt(
                                        " ",
                                        "cpu", #accelerator.device,
                                        1,
                                        do_classifier_free_guidance,
                                        negative_prompt=None,
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
    
    for i in range(len(text_list)):
        if random.random()<=args.uncaptioned_frac:
            text_list[i]=unconditioned_text.squeeze(0).clone().detach()

    def loss_fn(pred_embedding_batch:torch.Tensor, src_embedding_batch:torch.Tensor)->torch.Tensor:
        #pred_embedding_batch=embedding_util.embed_img_tensor(img_tensor_batch)
        return F.mse_loss(pred_embedding_batch.float(),src_embedding_batch.float(),reduce="mean")
    
    fake_image=torch.rand((1,3,args.image_size,args.image_size))
    fake_embedding=embedding_util.embed_img_tensor(fake_image)
    embedding_dim=fake_embedding.size()[-1]

    for component in [vae,text_encoder]:
        component.requires_grad_(False)
        component.to("cpu")
        #unet=unet.to(device,torch_dtype)
    
    unet.requires_grad_(False)

    accelerator.print("before ",pipeline.unet.config.sample_size)
    pipeline.unet.config.sample_size=args.image_size // pipeline.vae_scale_factor
    accelerator.print("after", pipeline.unet.config.sample_size)
    
    ratios=(args.train_split,(1.0-args.train_split)/2.0,(1.0-args.train_split)/2.0)
    accelerator.print('train/test/val',ratios)
    #batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
    embedding_list,test_embedding_list,val_embedding_list=split_list_by_ratio(embedding_list,ratios)

    
    #image_list= image_list #make_batches_same_size(image_list,args.batch_size)
    #text_list= text_list #[text_list[i:i + args.batch_size] for i in range(0, len(text_list), args.batch_size)]

    
    image_list,test_image_list,val_image_list=split_list_by_ratio(image_list,ratios)
    text_list,test_text_list,val_text_list=split_list_by_ratio(text_list,ratios)

    posterior_list,test_posterior_list,val_posterior_list=split_list_by_ratio(posterior_list,ratios)

    prompt_list,test_prompt_list,val_prompt_list=split_list_by_ratio(prompt_list,ratios)

    accelerator.print("prompt list",len(prompt_list))
    accelerator.print("image_list",len(image_list))
    accelerator.print("text_list",len(text_list))
    accelerator.print("posterior list",len(posterior_list))
    accelerator.print("embedding list",len(embedding_list))

    generic_dataset=load_dataset("jlbaker361/test_prompts",split="train")
    generic_tensor_list=[torch.from_numpy(np.array(row["text_embedding"])[0]) for row in generic_dataset]
    generic_str_list=[row["prompt"] for row in generic_dataset]

    for k in range(len(test_prompt_list)):
        #test_prompt_list[k]=generic_str_list[k%len(generic_str_list)]
        test_text_list[k]=generic_tensor_list[k%len(generic_str_list)]

    train_dataset=CustomDataset(image_list,embedding_list,text_list,posterior_list,prompt_list)
    val_dataset=CustomDataset(val_image_list,val_embedding_list,val_text_list,val_posterior_list,val_prompt_list)
    test_dataset=CustomDataset(test_image_list,test_embedding_list,test_text_list,test_posterior_list,test_prompt_list)

    for dataset_batch in train_dataset:
        break

    accelerator.print("dataset batch",type(dataset_batch))

    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size)
    test_loader=DataLoader(test_dataset,args.batch_size)


    for train_batch in train_loader:
        break
    for val_batch in val_loader:
        break

    accelerator.print("train batch",type(train_batch))
    accelerator.print("val batch",type(val_batch))
    '''for name, param in unet.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable shape {tuple(param.shape)}")'''




    if args.vanilla:
        unet=unet.to(device)

    #if args.training_type=="reward":
    vae=vae.to(unet.device)
    post_quant_conv=vae.post_quant_conv.to(unet.device)
    time_embedding=unet.time_embedding.to(unet.device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if args.fsdp:
        clip_model.logit_scale = torch.nn.Parameter(torch.tensor([clip_model.config.logit_scale_init_value]))
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    fid = FrechetInceptionDistance(feature=2048,normalize=True)
    accelerator.wait_for_everyone()
    clip_model,clip_processor,fid,unet,vae,post_quant_conv,scheduler,time_embedding=accelerator.prepare(clip_model,clip_processor,fid,unet,vae,post_quant_conv,scheduler,time_embedding)
    accelerator.wait_for_everyone()
    train_loader,test_loader,val_loader=accelerator.prepare(train_loader,test_loader,val_loader)
    accelerator.wait_for_everyone()
    #train_loader=accelerator.prepare_data_loader(train_loader,True)
    try:
        register_fsdp_forward_method(vae,"decode")
        accelerator.print("registered")
    except Exception as e:
        accelerator.print('register_fsdp_forward_method',e)
    vae.post_quant_conv=post_quant_conv
    unet.time_embedding=time_embedding
    pipeline.unet=unet
    pipeline.vae=vae

    for loader in [test_loader,val_loader,train_loader]:
        print(type(loader))

    

    def logging(data_loader,pipeline,num_inference_steps:int,baseline:bool=False,auto_log:bool=True,clip_model:CLIPModel=clip_model):
        metrics={}
        difference_list=[]
        embedding_difference_list=[]
        clip_alignment_list=[]
        image_list=[]
        fake_image_list=[]

        if args.pipeline=="lcm_post_lora":
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            pipeline.load_lora_weights(adapter_id)
            pipeline.fuse_lora()
        
        for b,batch in enumerate(data_loader):
            if args.vanilla:
                pipeline.vae=pipeline.vae.to(device)
                for k,v in batch.items():
                    if type(v)==torch.Tensor:
                        batch[k]=v.to(pipeline.unet.device,torch_dtype)
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            prompt_batch=batch["prompt"]
            do_denormalize= [True] * image_batch.size()[0]
            if len(image_batch.size())==3:
                image_batch=image_batch.unsqueeze(0)
                text_batch=text_batch.unsqueeze(0)
                embeds_batch=embeds_batch.unsqueeze(0)
            image_embeds=embeds_batch #.unsqueeze(0)
            if b==0:
                
                accelerator.print("testing","images",image_batch.size(),"text",text_batch.size(),"embeds",embeds_batch.size())
                accelerator.print("testing","images",image_batch.device,"text",text_batch.device,"embeds",embeds_batch.device)
            #image_batch=torch.clamp(image_batch, 0, 1)
            real_pil_image_set=pipeline.image_processor.postprocess(image_batch,"pil",do_denormalize)
            
            if baseline:
                #ip_adapter_image=F_v2.resize(image_batch, (224,224))
                fake_image=torch.stack([pipeline( num_inference_steps=num_inference_steps,prompt_embeds=text_batch,ip_adapter_image=ip_adapter_image,output_type="pt",height=args.image_size,width=args.image_size).images[0] for ip_adapter_image in real_pil_image_set])
            else:
                fake_image=pipeline(num_inference_steps=num_inference_steps,prompt_embeds=text_batch,ip_adapter_image_embeds=[image_embeds],output_type="pt",height=args.image_size,width=args.image_size).images
            
            #normal_image_set=pipeline(prompt_embeds=text_batch,output_type="pil").images
            image_batch=F_v2.resize(image_batch, (args.image_size,args.image_size))
            #print("img vs real img",fake_image.size(),image_batch.size())
            #image_embeds.to("cpu")
            image_batch=image_batch.to(fake_image.device)

            difference_list.append(F.mse_loss(fake_image,image_batch).cpu().detach().item())


            embedding_real=embedding_util.embed_img_tensor(image_batch)
            embedding_fake=embedding_util.embed_img_tensor(fake_image)
            embedding_difference_list.append(F.mse_loss(embedding_real,embedding_fake).cpu().detach().item())

            image_list.append(image_batch.cpu())
            fake_image_list.append(fake_image.cpu())
            
            
            
            pil_image_set=pipeline.image_processor.postprocess(fake_image,"pil",do_denormalize)
            


            
            inputs = clip_processor(
                text=prompt_batch, images=pil_image_set, return_tensors="pt", padding=True
            )
            for k,v in inputs.items():
                inputs[k]=v.to(clip_model.device)

            #outputs = clip_model(**inputs)
            #clip_text_embeds=outputs.text_embeds
            #clip_image_embeds=outputs.image_embeds
            #clip_difference=F.mse_loss(clip_image_embeds,clip_text_embeds)
            
            #clip_alignment_list.append(clip_difference.cpu().detach().item())

            for pil_image,real_pil_image,prompt in zip(pil_image_set,real_pil_image_set,prompt_batch):
                concat_image=concat_images_horizontally([real_pil_image,pil_image])
                metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(concat_image)
        #pipeline.scheduler =  DEISMultistepScheduler.from_config(pipeline.scheduler.config)


        if auto_log:
            accelerator.log(metrics)
        if args.pipeline=="lcm_post_lora":
            pipeline.unfuse_lora()
        return metrics
    
    for num_inference_steps in [2,4,10,20]:
        logging(test_loader,pipeline,num_inference_steps,True)

    training_start=time.time()
        

                

    


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