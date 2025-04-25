from diffusers.models.embeddings import IPAdapterFullImageProjection
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers import UNet2DConditionModel
import torch


def get_modules_of_types(model, target_classes):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, target_classes)]

def replace_ip_attn(unet:UNet2DConditionModel,in_features:int):
    layers=get_modules_of_types(unet,IPAdapterAttnProcessor2_0)
    for (name,module) in layers:
        out_features=module.to_k_ip[0].out_features
        new_k_ip=torch.nn.ModuleList([torch.nn.Linear(in_features,out_features)])
        setattr(module, "to_k_ip",new_k_ip)
        new_v_ip=torch.nn.ModuleList([torch.nn.Linear(in_features,out_features)])
        setattr(module, "to_v_ip",new_v_ip)