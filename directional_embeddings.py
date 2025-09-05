import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
from embedding_helpers import EmbeddingUtil
from directional_data import noun_list, action_list, location_list, style_list
import torch
from diffusers import StableDiffusionPipeline
from datasets import load_dataset, Dataset

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="directional")
parser.add_argument("--embedding",type=str,default="ssl")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/destination-embedding")
parser.add_argument("--src_dataset",type=str,default="jlbaker361/directional")
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--upload_interval",type=int,default=10)



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    image_processor=StableDiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").image_processor
    
    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    try:
        data_dict=load_dataset(args.dest_dataset,split="train").to_dict()
        start=len(data_dict["image"])
    except:
        data_dict={
        "image":[],
        "noun":[],
        "location":[],
        "action":[],
        "style":[],
        "seed":[],
        "prompt":[],
        "embedding":[]
        }
        start=0
    src_data=load_dataset(args.src_dataset,split="train")
    accelerator.print("starting at ",start)
    for k,row in enumerate(src_data):
        if k==args.limit:
            break
        if k<start:
            continue
        img_tensor= image_processor.preprocess(row["image"])
        embedding=embedding_util.embed_img_tensor(img_tensor)

        data_dict["embedding"].append(embedding)

        for key in data_dict.keys():
            if key != "embedding":
                data_dict[key].append(row[key])

        if k%args.upload_interval==0:
            try:
                Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)
            except:
                time.sleep(10)
                Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)

    time.sleep(10)
    Dataset.from_dict(data_dict).push_to_hub(args.dest_dataset)



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