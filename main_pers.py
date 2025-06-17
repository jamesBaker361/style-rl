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
from custom_scheduler import CompatibleFlowMatchEulerDiscreteScheduler

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
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,65))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)


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
    if accelerator.is_main_process:
        os.makedirs(save_dir,exist_ok=True)
    

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)


    adapter_id = "latent-consistency/lcm-lora-sdv1-5"
    if args.pipeline=="lcm":
        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
    elif args.pipeline=="lcm_post_lora":
        pipeline=DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        pipeline.load_lora_weights(adapter_id)
        pipeline.disable_lora()
    elif args.pipeline=="lcm_pre_lora":
        pipeline=DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        
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


    accelerator.print(pipeline.scheduler)
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
        

    accelerator.print(f"cross attention dim {embedding_dim} / {args.num_image_text_embeds} =  ",cross_attention_dim)
    if args.pipeline=="sana":
        prepare_ip_adapter(pipeline.transformer,accelerator.device,torch_dtype,cross_attention_dim)
    
    replace_ip_attn(denoising_model,
                    embedding_dim,
                    intermediate_embedding_dim,
                    cross_attention_dim,
                    args.num_image_text_embeds,
                    use_projection,args.identity_adapter)
    #print("image projection",unet.encoder_hid_proj.multi_ip_adapter.image_projection_layers[0])
    start_epoch=1
    persistent_loss_list=[]
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
            accelerator.print("loaded from  ",pretrained_weights_path)
        except Exception as e:
            accelerator.print("couldnt load from hf")
            accelerator.print(e)
    attn_layer_list=[p for (name,p ) in get_modules_of_types(denoising_model,IPAdapterAttnProcessor2_0)]
    attn_layer_list.append( denoising_model.encoder_hid_proj)
    accelerator.print("len attn_layers",len(attn_layer_list))
    for layer in attn_layer_list:
        layer.requires_grad_(True)


    accelerator.print("before ",denoising_model.config.sample_size)
    denoising_model.config.sample_size=args.image_size // pipeline.vae_scale_factor
    accelerator.print("after", denoising_model.config.sample_size)
    
    ratios=(args.train_split,(1.0-args.train_split)/2.0,(1.0-args.train_split)/2.0)
    accelerator.print('train/test/val',ratios)
    #batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
    embedding_list,test_embedding_list,val_embedding_list=split_list_by_ratio(embedding_list,ratios)
    
    image_list,test_image_list,val_image_list=split_list_by_ratio(image_list,ratios)
    text_list,test_text_list,val_text_list=split_list_by_ratio(text_list,ratios)

    posterior_list,test_posterior_list,val_posterior_list=split_list_by_ratio(posterior_list,ratios)

    prompt_list,test_prompt_list,val_prompt_list=split_list_by_ratio(prompt_list,ratios)

    accelerator.print("prompt list",len(prompt_list))
    accelerator.print("image_list",len(image_list))
    accelerator.print("text_list",len(text_list))
    accelerator.print("posterior list",len(posterior_list))
    accelerator.print("embedding list",len(embedding_list))

    if args.generic_test_prompts:
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

    params=list(set([p for p in denoising_model.parameters() if p.requires_grad]+[p for p in denoising_model.encoder_hid_proj.parameters() if p.requires_grad]))

    accelerator.print("trainable params: ",len(params))
    for i in range(accelerator.num_processes):
        if accelerator.process_index == i:
            print(f"Rank {i} checkpoint")
            torch.cuda.synchronize()
        accelerator.wait_for_everyone()

    optimizer=torch.optim.AdamW(params,lr=args.lr)

    if args.vanilla:
        denoising_model=denoising_model.to(device)

    #if args.training_type=="reward":
    vae=vae.to(denoising_model.device)
    
    #time_embedding=denoising_model.time_embedding.to(denoising_model.device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if args.fsdp:
        clip_model.logit_scale = torch.nn.Parameter(torch.tensor([clip_model.config.logit_scale_init_value]))
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    fid = FrechetInceptionDistance(feature=2048,normalize=True)
    accelerator.wait_for_everyone()
    clip_model,clip_processor,fid,denoising_model,vae,scheduler,optimizer=accelerator.prepare(clip_model,clip_processor,fid,denoising_model,vae,scheduler,optimizer)
    if hasattr(denoising_model,"post_quant_conv"):
        post_quant_conv=denoising_model.post_quant_conv.to(denoising_model.device)
        post_quant_conv=accelerator.prepare(post_quant_conv)
        vae.post_quant_conv=post_quant_conv
    if hasattr(denoising_model,"time_embedding"):
        time_embedding=denoising_model.time_embedding.to(denoising_model.device)
        time_embedding=accelerator.prepare(time_embedding)
        denoising_model.time_embedding=time_embedding

    if hasattr(denoising_model, "patch_embed"):
        patch_embed=denoising_model.patch_embed.to(denoising_model.device, torch_dtype)
        patch_embed=accelerator.prepare(patch_embed)
        denoising_model.patch_embed=patch_embed

    if hasattr(denoising_model,"caption_projection"):
        caption_projection=denoising_model.caption_projection.to(denoising_model.device, torch_dtype)
        caption_projection=accelerator.prepare(caption_projection)
        denoising_model.caption_projection=caption_projection

    if hasattr(denoising_model,"encoder_hid_proj"):
        encoder_hid_proj=denoising_model.encoder_hid_proj.to(denoising_model.device)
        encoder_hid_proj=accelerator.prepare(encoder_hid_proj)
        denoising_model.encoder_hid_proj=encoder_hid_proj
    accelerator.wait_for_everyone()
    train_loader,test_loader,val_loader=accelerator.prepare(train_loader,test_loader,val_loader)
    accelerator.wait_for_everyone()
    #train_loader=accelerator.prepare_data_loader(train_loader,True)
    try:
        register_fsdp_forward_method(vae,"decode")
        accelerator.print("registered")
    except Exception as e:
        accelerator.print('register_fsdp_forward_method',e)
    
    if args.pipeline=="sana":
        pipeline.transformer=denoising_model
    else:
        pipeline.unet=denoising_model
    pipeline.vae=vae

    for loader in [test_loader,val_loader,train_loader]:
        print(type(loader))

    

    def logging(data_loader,pipeline,baseline:bool=False,auto_log:bool=True,clip_model:CLIPModel=clip_model):
        metrics={}
        difference_list=[]
        embedding_difference_list=[]
        clip_alignment_list=[]
        image_list=[]
        fake_image_list=[]

        if args.pipeline=="sana":
            scheduler=SCMScheduler.from_config(scheduler.config)
            scheduler=accelerator.prepare(scheduler)
            pipeline.scheduler=scheduler
            accelerator.wait_for_everyone()

        if args.pipeline=="lcm_post_lora":
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            pipeline.enable_lora()
        pipeline=pipeline.to(accelerator.device)
        pipeline.vae=pipeline.vae.to(accelerator.device)
        pipeline.text_encoder=pipeline.text_encoder.to(accelerator.device)
        if hasattr(pipeline,"unet"):
            pipeline.unet.time_embedding=pipeline.unet.time_embedding.to(accelerator.device)
            pipeline.unet.time_embedding.linear_1.weight=pipeline.unet.time_embedding.linear_1.weight.to(accelerator.device)
            pipeline.unet.time_embedding.linear_1.bias=pipeline.unet.time_embedding.linear_1.bias.to(accelerator.device)

            pipeline.unet.time_embedding.linear_2.weight=pipeline.unet.time_embedding.linear_2.weight.to(accelerator.device)
            pipeline.unet.time_embedding.linear_2.bias=pipeline.unet.time_embedding.linear_2.bias.to(accelerator.device)

            if getattr(pipeline.unet.time_embedding, "cond_proj",None) is not None:
                pipeline.unet.time_embedding.cond_proj.weight=pipeline.unet.time_embedding.cond_proj.weight.to(accelerator.device)
        
        for b,batch in enumerate(data_loader):
            
            for k,v in batch.items():
                if type(v)==torch.Tensor:
                    batch[k]=v.to(accelerator.device,torch_dtype)
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            prompt_batch=batch["prompt"]
            
            if len(image_batch.size())==3:
                image_batch=image_batch.unsqueeze(0)
                text_batch=text_batch.unsqueeze(0)
                embeds_batch=embeds_batch.unsqueeze(0)
            batch_size=image_batch.size()[0]
            image_embeds=embeds_batch #.unsqueeze(0)
            do_denormalize= [True] * batch_size
            if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
                batched_negative_prompt_embeds=negative_text_embeds.expand((batch_size, -1,-1)).to(text_batch.device)
                negative_image_embeds=torch.zeros(image_embeds.size(),device=image_embeds.device)
                image_embeds=[torch.cat([negative_image_embeds,image_embeds],dim=0)]
            else:
                batched_negative_prompt_embeds=None
                image_embeds=[image_embeds]
            
            if b==0:
                if hasattr(pipeline,"unet"):
                    print("unet",pipeline.unet.device,"time embedding linear 1",pipeline.unet.time_embedding.linear_1.weight.device, )
                
                accelerator.print("testing","images",image_batch.size(),"text",text_batch.size(),"embeds",embeds_batch.size())
                accelerator.print("testing","images",image_batch.device,"text",text_batch.device,"embeds",embeds_batch.device)
                accelerator.print("testing","images",image_batch.dtype,"text",text_batch.dtype,"embeds",embeds_batch.dtype)
                if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
                    accelerator.print("testing","negstive",batched_negative_prompt_embeds.size(),batched_negative_prompt_embeds.device)
            #image_batch=torch.clamp(image_batch, 0, 1)
            real_pil_image_set=pipeline.image_processor.postprocess(image_batch,"pil",do_denormalize)
            
            if baseline:
                #ip_adapter_image=F_v2.resize(image_batch, (224,224))
                fake_image=torch.stack([pipeline( num_inference_steps=args.num_inference_steps,
                                                 prompt_embeds=text_batch,ip_adapter_image=ip_adapter_image, negative_prompt_embeds=batched_negative_prompt_embeds,
                                                 output_type="pt",height=args.image_size,width=args.image_size).images[0] for ip_adapter_image in real_pil_image_set])
            else:
                fake_image=pipeline(num_inference_steps=args.num_inference_steps,
                                    prompt_embeds=text_batch,ip_adapter_image_embeds=image_embeds,negative_prompt_embeds=batched_negative_prompt_embeds,
                                    output_type="pt",height=args.image_size,width=args.image_size).images
            
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

            outputs = clip_model(**inputs)
            clip_text_embeds=outputs.text_embeds
            clip_image_embeds=outputs.image_embeds
            clip_difference=F.mse_loss(clip_image_embeds,clip_text_embeds)
            
            clip_alignment_list.append(clip_difference.cpu().detach().item())

            for pil_image,real_pil_image,prompt in zip(pil_image_set,real_pil_image_set,prompt_batch):
                concat_image=concat_images_horizontally([real_pil_image,pil_image])
                metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(concat_image)
        #pipeline.scheduler =  DEISMultistepScheduler.from_config(pipeline.scheduler.config)
        metrics["difference"]=np.mean(difference_list)
        metrics["embedding_difference"]=np.mean(embedding_difference_list)
        metrics["text_alignment"]=np.mean(clip_alignment_list)
        #print("size",torch.cat(image_list).size())
        start=time.time()
        fid_dtype=next(fid.inception.parameters()).dtype
        fid_device=next(fid.inception.parameters()).device
        fid.update(torch.cat(image_list).to(fid_device,fid_dtype),real=True)
        fid.update(torch.cat(fake_image_list).to(fid_device,fid_dtype),real=False)
        metrics["fid"]=fid.compute().cpu().detach().item()
        end=time.time()
        print("fid elapsed ",end-start)
        if auto_log:
            accelerator.log(metrics)
        if args.pipeline=="lcm_post_lora":
            pipeline.disable_lora()
        return metrics

    training_start=time.time()
    
    accelerator.print(f"training from {start_epoch} to {args.epochs}")
    for e in range(start_epoch, args.epochs+1):
        scale=args.initial_scale+(float(e)/args.epochs)*(args.final_scale-args.initial_scale)
        accelerator.print("scale",scale)
        pipeline.set_ip_adapter_scale(scale)
        if e==args.reward_switch_epoch:
            args.training_type="reward"

        if args.pipeline=="sana":
            if args.training_type=="denoise":
                scheduler=CompatibleFlowMatchEulerDiscreteScheduler.from_config(scheduler.config)
            elif args.training_type!="denoise":
                scheduler=SCMScheduler.from_config(scheduler.config)
            scheduler=accelerator.prepare(scheduler)
            pipeline.scheduler=scheduler
            accelerator.wait_for_everyone()
        before_objects=find_cuda_objects()
        start=time.time()
        loss_buffer=[]
        for b,batch in enumerate(train_loader):

            for k,v in batch.items():
                if type(v)==torch.Tensor:
                    if args.deepspeed:
                        batch[k]=v.to(torch_dtype)
                    else:
                        batch[k]=v.to(device,torch_dtype)
                
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            posterior_batch=batch["posterior"]
            
            if e==start_epoch and b==0:
                print("text size",text_batch.size(),"embedding size",embeds_batch.size(),"img size",image_batch.size(),"latent size",posterior_batch.size())
                print("text device",text_batch.device,"embedding device",embeds_batch.device,"img device",image_batch.device,"latent device",posterior_batch.device)
                print("text ",text_batch.dtype,"embedding ",embeds_batch.dtype,"img ",image_batch.dtype,"latent ",posterior_batch.dtype)
            image_embeds=embeds_batch #.to(device) #.unsqueeze(1)
            #print('image_embeds',image_embeds.requires_grad,image_embeds.size())
            prompt=text_batch
            if args.epochs >1 and  random.random() <args.uncaptioned_frac:
                prompt=" "
            #print(pipeline.text_encoder)
            if args.training_type=="denoise":
                with accelerator.accumulate(params):
                    # Convert images to latent space
                    #if args.deepspeed:
                    if args.pipeline=="sana":
                        latents=posterior_batch / pipeline.scheduler.config.sigma_data
                    else:
                        latents = DiagonalGaussianDistribution(posterior_batch).sample()
                        
                    latents = latents * vae.config.scaling_factor



                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    
                    encoder_hidden_states = text_batch
                    batch_size=text_batch.size()[0]
                    #print("encoede hiiden states",encoder_hidden_states.requires_grad)
                    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
                    #timesteps = timesteps.long()

                    #https://github.com/NVlabs/Sana/blob/main/train_scripts/train_dreambooth_lora_sana.py
                        

                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        scheduler.register_to_config(prediction_type=args.prediction_type)

                    if scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif scheduler.config.prediction_type == "v_prediction":
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
                    
                    added_cond_kwargs={"image_embeds":[image_embeds]}
                    print('Image embeds size denoising',image_embeds.size())

                    if args.pipeline=="sana":
                        guidance = torch.full([1], 4.5, device=device, dtype=torch.float32)
                        guidance = guidance.expand(noisy_latents.shape[0]).to(noisy_latents.dtype)
                        guidance = guidance * denoising_model.config.guidance_embeds_scale
                        if args.vanilla:
                            with accelerator.autocast():
                                model_pred=compatible_forward_sana_transformer_model(
                                    denoising_model,
                                    noisy_latents,
                                    encoder_hidden_states=encoder_hidden_states,
                                    timestep=timesteps,return_dict=False,
                                    guidance=guidance,
                                    encoder_hid_proj=encoder_hid_proj,
                                    added_cond_kwargs=added_cond_kwargs
                                )[0]
                        else:
                            model_pred=compatible_forward_sana_transformer_model(
                                    denoising_model,
                                    noisy_latents,
                                    encoder_hidden_states=encoder_hidden_states,
                                    timestep=timesteps,return_dict=False,
                                    guidance=guidance,
                                    encoder_hid_proj=encoder_hid_proj,
                                    added_cond_kwargs=added_cond_kwargs
                                )[0]
                    else:
                        if args.vanilla:
                            with accelerator.autocast():
                                model_pred = denoising_model(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]
                                
                        else:
                            model_pred = denoising_model(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
            elif args.training_type=="reward":
                with accelerator.accumulate(params):
                    #latents = DiagonalGaussianDistribution(posterior_batch).sample()
                    if args.vanilla:
                        with accelerator.autocast():
                            images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                        #latents=latents, 
                                                        num_inference_steps=args.num_inference_steps, 
                                                        ip_adapter_image_embeds=[image_embeds],output_type="pt",truncated_backprop=False,reward_training=True,
                                                        use_resolution_binning=False,
                                                        height=args.image_size,width=args.image_size).images
                            #print("reward max, min",images.max(),images.min())
                            predicted=embedding_util.embed_img_tensor(images)
                            loss=loss_fn(predicted,embeds_batch)
                    else:
                        images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                        #latents=latents, 
                                                        num_inference_steps=args.num_inference_steps,
                                                          ip_adapter_image_embeds=[image_embeds],output_type="pt",
                                                          truncated_backprop=False,fsdp=True,reward_training=True,
                                                          use_resolution_binning=False,
                                                          height=args.image_size,width=args.image_size).images
                        predicted=embedding_util.embed_img_tensor(images)
                        loss=loss_fn(predicted,embeds_batch)
                    #loss=(loss-np.mean(loss_buffer))/np.std(loss_buffer)
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
            elif args.training_type=="latents_reward":
                with accelerator.accumulate(params):
                    latents = DiagonalGaussianDistribution(posterior_batch).sample()
                    if args.vanilla:
                            with accelerator.autocast():
                                images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                            #latents=latents, 
                                                            num_inference_steps=args.num_inference_steps, 
                                                            ip_adapter_image_embeds=[image_embeds],output_type="latent",truncated_backprop=False,reward_training=True,
                                                            height=args.image_size,width=args.image_size).images

                    else:
                        images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                        #latents=latents, 
                                                        num_inference_steps=args.num_inference_steps,
                                                            ip_adapter_image_embeds=[image_embeds],output_type="latent",
                                                            truncated_backprop=False,fsdp=True,reward_training=True,
                                                            height=args.image_size,width=args.image_size).images
                    loss=loss_fn(images,latents)
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
            elif args.training_type=="mse_reward":
                batch_size=image_batch.size()[-1]
                image_batch=pipeline.image_processor.postprocess(image_batch,"pt",[True]*batch_size)
                with accelerator.accumulate(params):
                    #latents = DiagonalGaussianDistribution(posterior_batch).sample()
                    if args.vanilla:
                        with accelerator.autocast():
                            images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                        #latents=latents, 
                                                        num_inference_steps=args.num_inference_steps, 
                                                        ip_adapter_image_embeds=[image_embeds],output_type="pt",truncated_backprop=False,reward_training=True,
                                                        use_resolution_binning=False,
                                                        height=args.image_size,width=args.image_size).images
                            #print("reward max, min",images.max(),images.min())
                            loss=loss_fn(image_batch,images)
                    else:
                        images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                        #latents=latents, 
                                                        num_inference_steps=args.num_inference_steps,
                                                          ip_adapter_image_embeds=[image_embeds],output_type="pt",
                                                          truncated_backprop=False,fsdp=True,reward_training=True,
                                                          use_resolution_binning=False,
                                                          height=args.image_size,width=args.image_size).images
                        loss=loss_fn(image_batch,images)
                    #loss=(loss-np.mean(loss_buffer))/np.std(loss_buffer)
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
            loss_buffer.append(loss.cpu().detach().item())
            if torch.cuda.is_available():
                before_memory=get_gpu_memory_usage()["allocated_mb"]
            after_objects=find_cuda_objects()
            len_after=len(after_objects)
            delete_unique_objects(before_objects,after_objects)
            after_after_objects=find_cuda_objects()
            #print("deleted ",len_after-len(after_after_objects))
            if torch.cuda.is_available():
                after_memory=get_gpu_memory_usage()["allocated_mb"]
                #print(f"freed {before_memory-after_memory} mb")

        end=time.time()
        elapsed=end-start
        accelerator.print(f"\t epoch {e} elapsed {end-start}")
        accelerator.log({
            "loss_mean":np.mean(loss_buffer),
            "loss_std":np.std(loss_buffer),
            "elapsed":elapsed
        })
        persistent_loss_list.append(np.mean(loss_buffer))
        torch.cuda.empty_cache()
        accelerator.free_memory()
        if e%args.validation_interval==0:
            before_objects=find_cuda_objects()
            with torch.no_grad():

                start=time.time()
                clip_model=clip_model.to(pipeline.unet.device)
                val_metrics=logging(val_loader,pipeline,clip_model=clip_model)
                clip_model=clip_model.cpu()
                end=time.time()
                accelerator.print(f"\t validation epoch {e} elapsed {end-start}")
                persistent_fid_list.append(val_metrics["fid"])
                persistent_text_alignment_list.append(val_metrics["text_alignment"])
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)
        if e%args.upload_interval==0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                state_dict={name: param for name, param in pipeline.unet.named_parameters() if param.requires_grad}
                print("state dict len",len(state_dict))
                torch.save(state_dict,save_path)
                with open(config_path,"w+") as config_file:
                    data={"start_epoch":e,
                        "persistent_loss_list":persistent_loss_list,
                        "persistent_text_alignment_list":persistent_text_alignment_list,
                        "persistent_fid_list":persistent_fid_list}
                    json.dump(data,config_file, indent=4)
                    pad = " " * 1024  # ~1KB of padding
                    config_file.write(pad)
                accelerator.print(f"saved {save_path}")
                api.upload_file(path_or_fileobj=save_path,
                                path_in_repo=WEIGHTS_NAME,
                                repo_id=args.name)
                api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                repo_id=args.name)
                accelerator.print(f"uploaded {args.name} to hub")
    training_end=time.time()
    accelerator.print(f"total trainign time = {training_end-training_start}")
    accelerator.free_memory()
    clip_model=clip_model.to(pipeline.unet.device)
    metrics=logging(test_loader,pipeline,auto_log=False)
    new_metrics={}
    for k,v in metrics.items():
        new_metrics["test_"+k]=v
        accelerator.print("\tTEST",k,v)
    accelerator.log(new_metrics)

    if args.pipeline!="sana":
        if args.pipeline=="lcm":
            baseline_pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        else:
            baseline_pipeline=DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device,torch_dtype=torch_dtype)
            
            baseline_pipeline.load_lora_weights(adapter_id)
            baseline_pipeline.fuse_lora()
        baseline_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        try:
            baseline_pipeline.safety_checker=None
        except Exception as err:
            accelerator.print("tried to set safety checker to None",err)
        b_unet=baseline_pipeline.unet.to(device,torch_dtype)
        b_text_encoder=baseline_pipeline.text_encoder.to(device,torch_dtype)
        b_vae=baseline_pipeline.vae.to(device,torch_dtype)
        b_image_encoder=baseline_pipeline.image_encoder.to(device,torch_dtype)

        b_unet,b_text_encoder,b_vae,b_image_encoder=accelerator.prepare(b_unet,b_text_encoder,b_vae,b_image_encoder)
        baseline_pipeline.unet=b_unet
        baseline_pipeline.text_encoder=b_text_encoder
        baseline_pipeline.vae=b_vae
        baseline_pipeline.image_encoder=b_image_encoder
        baseline_metrics=logging(test_loader,baseline_pipeline,baseline=True)
        new_metrics={}
        for k,v in baseline_metrics.items():
            new_metrics["baseline_"+k]=v
            accelerator.print("\tBASELINE",k,v)
        accelerator.log(new_metrics)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict={name: param for name, param in pipeline.unet.named_parameters() if param.requires_grad}
        print("state dict len",len(state_dict))
        '''for k in state_dict.keys():
            print("\t",k)'''
        torch.save(state_dict,save_path)
        with open(config_path,"w+") as config_file:
            data={"start_epoch":args.epochs+1,
                        "persistent_loss_list":persistent_loss_list,
                        "persistent_text_alignment_list":persistent_text_alignment_list,
                        "persistent_fid_list":persistent_fid_list}
            json.dump(data,config_file, indent=4)
            pad = " " * 1024  # ~1KB of padding
            config_file.write(pad)
        print(f"saved {save_path}")
        api.upload_file(path_or_fileobj=save_path,
                                path_in_repo=WEIGHTS_NAME,
                                repo_id=args.name)
        api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                repo_id=args.name)
        print(f"uploaded {args.name} to hub")
        for k in ["persistent_loss_list","persistent_text_alignment_list","persistent_fid_list"]:
            persistent_list=data[k]
            key=k[:-5]
            for value in persistent_list:
                accelerator.log({key:value})
    accelerator.log({"finished":True})
        

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