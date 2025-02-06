from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
from diffusers import DiffusionPipeline

class KeywordDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self,sd_pipeline:DiffusionPipeline,keyword:str="",use_lora:bool=True):
        self.sd_pipeline=sd_pipeline
        self.keyword=keyword
        self.use_lora=use_lora

    def get_trainable_layers(self):
        return  [
        p for name, p in self.sd_pipeline.unet.named_parameters()
        if p.requires_grad and self.keyword in name
            ]