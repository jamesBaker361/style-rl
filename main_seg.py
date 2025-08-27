import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from ipattn import MonkeyIPAttnProcessor, get_modules_of_types,reset_monkey
import torch.nn.functional as F
import math
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention
from diffusers.utils import deprecate, is_torch_xla_available, logging
from typing import Optional,List
from diffusers.image_processor import IPAdapterMaskProcessor
import torch
from main_pers import concat_images_horizontally
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.transforms.functional import to_pil_image

from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from custom_sam_detector import CustomSamDetector
import datasets
import wandb

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="seg-ip")
parser.add_argument("--load_hf",action="store_true",help="whether to load a special pretrained model")
parser.add_argument("--embedding",type=str, help="ignore unless load from hf; its the embedding type for embedding helpers")
parser.add_argument("--pretrained_model_path",type=str,default="")
parser.add_argument("--dataset",type=str, default="jlbaker361/dino-art_coco_captioned-50-sana")
parser.add_argument("--use_test_split",action="store_true", help="only true for league dataset")
parser.add_argument("--initial_steps",type=int,default=4,help="how many steps for the initial inference")
parser.add_argument("--initial_mask_step_list",nargs="*",help="steps to generate mask from",type=int)
parser.add_argument("--final_steps",type=int,default=8, help="how many steps for final inference (with mask)")
parser.add_argument("--final_mask_steps_list",nargs="*",help="steps to apply mask from",type=int)
parser.add_argument("--final_adapter_steps_list",nargs="*",help="steps to apply adapter for (regardless of mask)",type=int)
parser.add_argument("--threshold",type=float,default=0.5,help="threshold for mask")
parser.add_argument("--limit",type=int,default=-1,help="limit of samples")
parser.add_argument("--layer_index",type=int,default=15)
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--token",type=int,default=1, help="which IP token is attention")
parser.add_argument("--overlap_frac",type=int,default=0.8)
parser.add_argument("--segmentation_attention_method",type=str,default="overlap or exclusive")

def get_mask(layer_index:int, 
             attn_list:list,step:int,
             token:int,dim:int,
             threshold:float,
             kv_type:str="ip",
             vae_scale:int=8):
    #print("layer",layer_index)
    module=attn_list[layer_index][1] #get the module no name
    #module.processor.kv_ip
    if kv_type=="ip":
        processor_kv=module.processor.kv_ip
    elif kv_type=="str":
        processor_kv=module.processor.kv
    size=processor_kv[step].size()
    #print('\tprocessor_kv[step].size()',processor_kv[step].size())
    
    avg=processor_kv[step].mean(dim=1).squeeze(0)
    #print("\t avg ", avg.size())
    latent_dim=int (math.sqrt(avg.size()[0]))
    #print("\tlatent",latent_dim)
    avg=avg.view([latent_dim,latent_dim,-1])
    #print("\t avg ", avg.size())
    avg=avg[:,:,token]
    #print("\t avg ", avg.size())
    avg_min,avg_max=avg.min(),avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min)  # [0,1]
    x_norm[x_norm < threshold]=0.
    avg = (x_norm * 255)
    #avg=F.interpolate(avg.unsqueeze(0).unsqueeze(0), size=(dim, dim), mode="nearest").squeeze(0).squeeze(0)

    return avg

def main(args):
    with torch.no_grad():
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
        accelerator.init_trackers(project_name=args.project_name,config=vars(args))

        custom_sam= CustomSamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints").to(accelerator.device)

        pipe = StableDiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float16,
        ).to(accelerator.device)

        # Load IP-Adapter
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(0.5)

        setattr(pipe,"safety_checker",None)

        #if args.load_hf:
            

        

        attn_list=get_modules_of_types(pipe.unet,Attention)

        for [name,_] in attn_list:
            print(name)

        for name,module in attn_list:
            if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==IPAdapterAttnProcessor2_0:
                setattr(module,"processor",MonkeyIPAttnProcessor(module.processor,name))



        #monkey_attn_list=get_modules_of_types(pipe.unet,MonkeyIPAttnProcessor)

        data=datasets.load_dataset(args.dataset)
        data=data["train"]

        for k,row in enumerate(data):
            if k==args.limit:
                for index,[name,module] in enumerate(attn_list):
                    
                    if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==MonkeyIPAttnProcessor:
                        #print(index,name,type(module),type(module.processor))
                        mask=sum([get_mask(index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
                        print(index,name,mask.size())
                break
            reset_monkey(pipe)
            ip_adapter_image=row["image"]
            prompt="eating ice cream"
            generator=torch.Generator()
            generator.manual_seed(123)
            pipe.set_ip_adapter_scale(0.5)
            initial_image=pipe(prompt,args.dim,args.dim,args.initial_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]

            mask=sum([get_mask(args.layer_index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
            tiny_mask=mask.clone()
            tiny_mask_pil=to_pil_image(1-tiny_mask)
            #print("mask size",mask.size())

            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

            

            mask_pil=to_pil_image(1-mask)
            color_rgba = initial_image.convert("RGB")
            mask_pil = mask_pil.convert("RGB")  # must be single channel for alpha

            #print(mask.size,color_rgba.size)

            # Apply as alpha (translucent mask)
            masked_img=Image.blend(color_rgba, mask_pil, 0.5)

            mask[mask>1]=1.
            #mask=1-mask
            #print(mask.size())
            mask_processor = IPAdapterMaskProcessor()
            #print(mask_processor.config)
            mask = mask_processor.preprocess(mask)
            #print(mask.size())
            #print("mask size",mask.size())

            masked_list=[]
            for index,[name,module] in enumerate(attn_list):
                if getattr(module,"processor",None)!=None and type(getattr(module,"processor",None))==MonkeyIPAttnProcessor:
                    _mask=sum([get_mask(index,attn_list,step,args.token,args.dim,args.threshold) for step in args.initial_mask_step_list])
                    _mask=F.interpolate(_mask.unsqueeze(0).unsqueeze(0), size=(args.dim, args.dim), mode="nearest").squeeze(0).squeeze(0)

                

                    ''''bw_img = Image.fromarray(_mask.cpu().numpy(), mode="L")  # "L" = 8-bit grayscale
                    _mask_pil = ImageOps.invert(bw_img)'''
                    color_rgba = initial_image.convert("RGB")
                    _mask_pil = to_pil_image(1-_mask).convert("RGB")  # must be single channel for alpha

                    #print(_mask.size(),_mask_pil.size,color_rgba.size)

                    # Apply as alpha (translucent mask)
                    _masked_img=Image.blend(color_rgba, _mask_pil, 0.5)

                    masked_list.append(_masked_img)

            first_concat=concat_images_horizontally(masked_list)

            accelerator.log({
                "first_concat":wandb.Image(first_concat)
            })

            generator=torch.Generator()
            generator.manual_seed(123)
            mask_step_list=args.final_mask_steps_list        
            scale_step_dict={i:0  for i in range(args.final_steps) }
            for k in args.final_adapter_steps_list:
                scale_step_dict[k]=1.0
            final_image=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image,generator=generator,cross_attention_kwargs={
                "ip_adapter_masks":mask
            }, mask_step_list=mask_step_list,scale_step_dict=scale_step_dict).images[0]

            generator=torch.Generator()
            generator.manual_seed(123)
            final_image_unmasked=pipe(prompt,args.dim,args.dim,args.final_steps,ip_adapter_image=ip_adapter_image,generator=generator).images[0]
            torch.cuda.empty_cache()
            
            segmented_image,map_list=custom_sam(initial_image,detect_resolution=args.dim)
            accelerator.log({
                "segmented":wandb.Image(segmented_image)
            })
            mask=mask.cpu()

            if args.segmentation_attention_method=="exclusive":
                map_mask=torch.ones((args.dim,args.dim))
            elif args.segmentation_attention_method=="overlap":
                map_mask=torch.zeros((args.dim,args.dim))
            for ann in map_list:
                map_=ann["segmentation"]

                map_=torch.from_numpy(map_).cpu()
                #map_=F.interpolate(map_.unsqueeze(0).unsqueeze(0), (args.dim,args.dim)).squeeze(0).squeeze(0)

                if args.segmentation_attention_method=="exclusive":
                    merged=map_*mask

                    map_mask=merged*map_mask

                elif args.segmentation_attention_method=="overlap":
                    print("\tmap mask size", map_mask.size())
                    n_ones=map_.sum()
                    merged=map_*mask
                    if merged.sum()>= args.overlap_frac * n_ones:
                        map_mask=torch.max(map_,map_mask)

            print("map mask size", map_mask.size())
            map_mask_pil=to_pil_image(1-map_mask).convert("RGB")

            
            concat=concat_images_horizontally([ip_adapter_image.resize([args.dim,args.dim],0),mask_pil,map_mask_pil,masked_img, segmented_image,initial_image,final_image,final_image_unmasked])
            accelerator.log({
                "image": wandb.Image(concat)
            })
            accelerator.log({"tiny_mask":wandb.Image(tiny_mask_pil)})



        




    return

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