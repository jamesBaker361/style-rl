from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
from diffusers import DiffusionPipeline,LatentConsistencyModelPipeline

class KeywordDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self,sd_pipeline:DiffusionPipeline,keyword:str="",use_lora:bool=False):
        self.sd_pipeline=sd_pipeline
        self.keyword=keyword
        self.use_lora=use_lora

    def get_trainable_layers(self):
        return  [
        p for name, p in self.sd_pipeline.unet.named_parameters()
        if p.requires_grad and self.keyword in name
            ]
    

class CompatibleLatentConsistencyModelPipeline(LatentConsistencyModelPipeline):
    def check_inputs(self, prompt,
                      height, 
                      width, 
                      callback_steps, 
                      negative_prompt=None,
                      prompt_embeds = None, 
                      negative_prompt_embeds=None,
                      ip_adapter_image=None, 
                      ip_adapter_image_embeds=None, 
                      callback_on_step_end_tensor_inputs=None):
        return super().check_inputs(prompt, height, width, callback_steps, prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, callback_on_step_end_tensor_inputs)