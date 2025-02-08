from diffusers import UNet2DConditionModel

import torch
import torch.nn as nn
from peft.tuners.lora import LoraModel

class LoRAAttention(nn.Module):
    def __init__(self, base_layer, rank, keyword):
        super().__init__()
        self.base_layer = base_layer  # Store original Linear layer
        self.rank = rank
        self.keyword=keyword
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Register LoRA matrices on the same device
        self.register_parameter(f"{keyword}_A", nn.Parameter(torch.randn(base_layer.weight.shape[0], rank, device=device,dtype=dtype) * 0.01,requires_grad=True))
        self.register_parameter(f"{keyword}_B", nn.Parameter(torch.randn(rank, base_layer.weight.shape[1], device=device,dtype=dtype) * 0.01,requires_grad=True))

    def forward(self, x):
        delta_W = getattr(self, f"{self.keyword}_A") @ getattr(self, f"{self.keyword}_B")  # LoRA weight update
        return nn.functional.linear(x, self.base_layer.weight + delta_W, self.base_layer.bias)


def assign_value_to_attr(obj, attr_path, value):
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:  # Traverse all but the last attribute
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def apply_lora(unet:UNet2DConditionModel,down_block_indices:list,up_block_indices:list,use_mid_block:bool,rank=4,keyword:str="lora")-> UNet2DConditionModel:
    blocks=[block for i,block in enumerate(unet.down_blocks) if i in down_block_indices]+[block for i,block in enumerate(unet.up_blocks) if i in up_block_indices]
    if use_mid_block:
        blocks.append(unet.mid_block)
    
    for block_idx, block in enumerate(blocks):
        print(type(block))
        lora_target_layers = []
        for name, module in block.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name for x in ["to_q", "to_k", "to_v", "to_out.0"]):
                lora_target_layers.append(name)
                #print(f"adding  {name} to dict")
        
        for layer_name in lora_target_layers:
            module = dict(block.named_modules())[layer_name]  # Get the module reference
            new_module=LoRAAttention(module, rank=rank,keyword=keyword)
            setattr(block, layer_name, new_module)
            assign_value_to_attr(block,layer_name,new_module)
                #module = dict(block.named_modules())[layer_name]
                #print(module.keyword)
                #print(f"🔹 Replacing {name} with LoRA-attention")

    return unet
