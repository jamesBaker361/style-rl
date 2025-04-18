import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
import time

parser=argparse.ArgumentParser()

def main(args):
    pass

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