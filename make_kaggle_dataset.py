import kagglehub
import os
from PIL import Image

from datasets import Dataset,load_dataset
from PIL import Image
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
import time
from PIL import Image
import random
import torch
from accelerate import Accelerator
import time
from datasets import Dataset
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from pipelines import CompatibleLatentConsistencyModelPipeline
from sana_pipelines import CompatibleSanaSprintPipeline
from diffusers import StableDiffusionPipeline

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil
from custom_vae import public_encode
from gpu_helpers import find_cuda_objects, delete_unique_objects


parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="asahi417/wikiart-face")
parser.add_argument("--output_dataset",type=str,default="jlbaker361/test-wikiface")
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--rewrite",action="store_true")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--pipeline",type=str,default="sana",help="sana or lcm")

def main(args):

    # Download latest version
    path = kagglehub.dataset_download("kaustubhdhote/human-faces-dataset")

    print("Path to dataset files:", path)
    real_images="Real Images"

    real_dir=os.path.join(path,"Human Faces Dataset",real_images)

    jpg_files = [f for f in os.listdir(real_dir) if f.lower().endswith('.jpg')]
    print(jpg_files[0])



    print("len",len(jpg_files))

    raw_data=[Image.open(os.path.join(real_dir, file)) for file in jpg_files]

    random.seed(42)
    with torch.no_grad():
        
        accelerator=Accelerator(mixed_precision=args.mixed_precision)
        with accelerator.autocast():
            torch_dtype={
                "fp16":torch.float16,
                "no":torch.float32
            }[args.mixed_precision]
            device=accelerator.device

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device,torch_dtype)

            raw_data=[Image.open(os.path.join(real_dir, file)) for file in jpg_files]
            random.shuffle(raw_data)

            embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

            new_dataset={
                "image":[],
                "embedding":[],
                "text":[],
                "prompt":[],
                "posterior":[],
                "attention_mask":[]
            }

            try:
                old_data=load_dataset(args.output_dataset)
                existing=True
                len_old=len([r for r in old_data])
            except:
                existing=False

            if args.pipeline=="lcm":
                pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
            elif args.pipeline=="sana":
                pipeline=CompatibleSanaSprintPipeline.from_pretrained("Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",device=accelerator.device)
                pipeline.do_classifier_free_guidance=False
            elif args.pipeline=="stability":
                pipeline=StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1",device=accelerator.device)
                
            pipeline=pipeline.to(accelerator.device)
            pipeline.text_encoder=pipeline.text_encoder.to(accelerator.device,torch_dtype)
            pipeline.vae=pipeline.vae.to(accelerator.device,torch_dtype)
            pipeline,model=accelerator.prepare(pipeline,model)

            before=find_cuda_objects()
            for k,row in enumerate(raw_data):
                print(k)
                if k==args.limit:
                    break
                image=row["image"].convert("RGB").resize((256,256))

                blip_inputs = processor(image, return_tensors="pt").to(device,torch_dtype)

                out = model.generate(**blip_inputs)
                text=processor.decode(out[0], skip_special_tokens=True)
                prompt=text
                encoded_text_attention_mask=-1
                if args.pipeline=="lcm":
                    encoded_text, _ = pipeline.encode_prompt(
                                                    prompt=text,
                                                    device=accelerator.device,
                                                    num_images_per_prompt=1,
                                            )
                elif args.pipeline=="sana":
                    encoded_text, encoded_text_attention_mask = pipeline.encode_prompt(
                                                    prompt=text,
                                                    device=accelerator.device,
                                                    num_images_per_prompt=1,
                                            )
                    
                elif args.pipeline=="stability":
                    encoded_text, _ = pipeline.encode_prompt(
                                                    prompt=text,
                                                    device=accelerator.device,
                                                    num_images_per_prompt=1,
                                                    do_classifier_free_guidance=False
                                            )
                new_dataset["attention_mask"].append(encoded_text_attention_mask)
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0).cpu().detach().numpy()
                new_dataset["image"].append(image)
                new_dataset["embedding"].append(embedding)
                new_dataset["text"].append(encoded_text)
                
                new_dataset["prompt"].append(prompt)
                image=pipeline.image_processor.preprocess(image)
                
                posterior=public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).squeeze(0).cpu().detach().numpy()
                if k==0:
                    posterior=public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).squeeze(0).cpu().detach().numpy()
                    print("image max min",image.max(),image.min())
                    print("post min max",public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).max(),public_encode(pipeline.vae,image.to(accelerator.device,torch_dtype)).min())
                    print("posterior",posterior)
                new_dataset["posterior"].append(posterior)
                torch.cuda.empty_cache()
                after=find_cuda_objects()
                delete_unique_objects(before,after)
                if k+1 %500==0:
                    print("processed: ",k+1)
                    if existing==False or args.rewrite:
                        time.sleep(random.randint(1,10))
                        Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)
                    else:
                        if k+1> len_old:
                            time.sleep(random.randint(1,10))
                            Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)




            time.sleep(random.randint(1,10))
            Dataset.from_dict(new_dataset).push_to_hub(args.output_dataset)

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
