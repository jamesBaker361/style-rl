from accelerate import Accelerator
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
from huggingface_hub import PyTorchModelHubMixin
from typing import Union,Any,Optional,Callable,List,Dict
import torch
import torch.nn.functional as F
import random
from PIL import Image
import wandb
import numpy as np

class HookTrainer(PyTorchModelHubMixin):
    def __init__(self,accelerator:Accelerator,
                 epochs:int,
                 num_inference_steps:int,
                 gradient_accumulation_steps:int,
                 batches_per_epoch:int,
                 ddpo_pipeline:DDPOStableDiffusionPipeline,
                 prompt_fn:Callable,
                 image_size:int,
                 target_activations:dict,
                 keyword:str="",
                 train_learning_rate:float=1e-3,
                 train_adam_beta1:float=0.9,
                 train_adam_beta2:float=0.999,
                 train_adam_weight_decay:float=1e-4,
                 train_adam_epsilon:float=1e-8):
        self.accelerator=accelerator
        self.epochs=epochs
        self.num_inference_steps=num_inference_steps
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.batches_per_epoch=batches_per_epoch
        self.ddpo_pipeline=ddpo_pipeline
        self.prompt_fn=prompt_fn
        self.image_size=image_size
        self.target_activations=target_activations
        self.keyword=keyword
        self.train_learning_rate=train_learning_rate
        self.train_adam_beta1=train_adam_beta1
        self.train_adam_beta2=train_adam_beta2
        self.train_adam_weight_decay=train_adam_weight_decay
        self.train_adam_epsilon=train_adam_epsilon

    def _setup_optimizer(self, trainable_layers_parameters):
        if self.config.train_use_8bit_adam:
            import bitsandbytes

            optimizer_cls = bitsandbytes.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            trainable_layers_parameters,
            lr=self.train_learning_rate,
            betas=(self.train_adam_beta1, self.train_adam_beta2),
            weight_decay=self.train_adam_weight_decay,
            eps=self.train_adam_epsilon,
        )
    
    def save_image(self,image:Image.Image):
        self.accelerator.log({
            f"{self.keyword}_training_image":wandb.Image(image)
        })

    def train(self,*args,**kwargs):
        sd_pipeline=self.ddpo_pipeline.sd_pipeline
        unet=sd_pipeline.unet
        hooks=[]
        activations = {}

            # Hook function
        def hook_fn(module, input, output):
            activations[module] = output # Store activation output

        # Register hooks for all encoder and decoder blocks
        
        for layer in self.target_activations.keys():
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)  # Keep track of hooks for later removal

        print(f"Registered {len(hooks)} hooks.")

        trainable_layers = self.sd_pipeline.get_trainable_layers()

        optimizer=self._setup_optimizer(
            trainable_layers.parameters() if not isinstance(trainable_layers, list) else trainable_layers
        )

        optimizer=self.accelerator.prepare(optimizer)

        for e in range(self.epochs):
            loss_list=[]
            with self.accelerator.accumulate():
                for step in range(self.gradient_accumulation_steps):
                    prompt=self.prompt_fn()
                    image=sd_pipeline(prompt,num_inference_steps=self.num_inference_steps,height=self.image_size,width=self.image_size).images[0]
                    key = random.choice(list(activations.keys()))
                    value=activations[key]
                    loss=F.mse_loss(self.target_activations[key],value)
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_list.append(loss.detach().cpu().numpy())
                self.save_image(image)
            avg_loss=np.mean(loss_list)
            self.accelerator.log({f"{self.keyword}_loss":avg_loss})
            self.accelerator.log({f"all_loss":avg_loss})

        for hook in hooks:
            hook.remove()

        self.accelerator.free_memory()

        
        
                
                