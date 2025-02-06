import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel,CLIPTokenizer
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from datasets import load_dataset
import torch
import time

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="style_creative")
parser.add_argument("--prompt",type=str,default="wizard")
parser.add_argument("--style_dataset",type=str,default="jlbaker361/stylization")
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--limit",type=int,default=5)
parser.add_argument("--method",type=str,default="ddpo")
parser.add_argument("--image_dim",type=int,default=256)
parser.add_argument("--num_inference_steps",type=int,default=4)

def get_vit_embeddings(vit_processor: ViTImageProcessor, vit_model: BetterViTModel, image_list:list,return_numpy:bool=True):
    '''
    returns (vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list)
    '''
    vit_embedding_list=[]
    vit_content_embedding_list=[]
    vit_style_embedding_list=[]
    for image in image_list:
        vit_inputs = vit_processor(images=[image], return_tensors="pt")
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



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16
    }[args.mixed_precision]


    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
    pipe.to(torch_device="cuda", torch_dtype=torch_dtype)

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    try:
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
        vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16')
    except:
    
        vit_processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16',force_download=True)
        vit_model = BetterViTModel.from_pretrained('facebook/dino-vitb16',force_download=True)
    vit_model.eval()
    vit_model.requires_grad_(False)

    # Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
    num_inference_steps = args.num_inference_steps
    #images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0,height=args.image_size,width=args.image_size).images
    #images[0].save("image.png")
    data=load_dataset(args.style_dataset,split="train")
    for i, row in enumerate(data):
        if i<args.start or i>=args.limit:
            continue
        label=row["label"]
        images=[row[f"image_{k}"] for k in range(4)]
        vit_embedding_list,vit_style_embedding_list, vit_content_embedding_list=get_vit_embeddings(vit_processor,vit_model,images,False)
        print(type(vit_style_embedding_list[0]))
        try:
            print("size",vit_style_embedding_list[0].size)
        except:
            try:
                print("shape",vit_style_embedding_list[0].shape)
            except:
                try:
                    print("len",len(vit_style_embedding_list[0]))
                except:
                    pass

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