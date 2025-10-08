from directional_data import noun_list, action_list, location_list, style_list
from datasets import Dataset,load_dataset
import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np




def get_paired_row(row:dict,attribute:str,dataset:Dataset)->dict:
    target=row[attribute]
    other_attributes=["noun","location","action","style","seed"]
    for att in other_attributes:
        print(row[att],end=",")
    other_attributes.remove(attribute)

    def valid(x_row):
        if x_row[attribute].strip()!="":
            return False
        for att in other_attributes:
            if x_row[att]!=row[att]:
                return False
            

        return True

    paired_row=dataset.filter(lambda x_row:
                       valid(x_row)
                       )
    return next(iter(paired_row))

def get_average_embedding(attribute:str, target:str,dataset:Dataset)-> list:
    target_dataset=dataset.filter(
        lambda row: row[attribute]==target
    )
    difference_list=[]

    for target_row in target_dataset:
        paired_row=get_paired_row(target_row,attribute,dataset)
        target_embedding=np.array(target_row["embedding"])
        base_embedding=np.array(paired_row["embedding"])
        difference_list.append(target_embedding-base_embedding)

    return np.mean(difference_list)



parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="evaluation-creative")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    #accelerator.init_trackers(project_name=args.project_name,config=vars(args))

    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    data=load_dataset("jlbaker361/directional-90-ssl",split="train")

    for row in data:
        break

    output_dict={
        "attribute":[],
        "value":[],
        "embedding":[]
    }

    for attribute, value_list in zip(["location","action","style"], [location_list,action_list,style_list]):
        for value in value_list:
            output_dict["attribute"].append(attribute)
            output_dict["value"].append(value)
            output_dict["embedding"].append(get_average_embedding(attribute,value,data))

    Dataset.from_dict(output_dict)

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