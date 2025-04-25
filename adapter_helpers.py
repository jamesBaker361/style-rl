from diffusers.models.embeddings import IPAdapterFullImageProjection,MultiIPAdapterImageProjection,ImageProjection
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers import UNet2DConditionModel
import torch


def get_modules_of_types(model, target_classes):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, target_classes)]

def replace_ip_attn(unet:UNet2DConditionModel,cross_attention_dim:int,embedding_size:int):
    layers=get_modules_of_types(unet,IPAdapterAttnProcessor2_0)
    for (name,module) in layers:
        out_features=module.to_k_ip[0].out_features
        new_k_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features)])
        new_k_ip.to(unet.device)
        setattr(module, "to_k_ip",new_k_ip)
        new_v_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features)])
        new_v_ip.to(unet.device)
        setattr(module, "to_v_ip",new_v_ip)

    unet.encoder_hid_proj=MultiIPAdapterImageProjection([ImageProjection(embedding_size,cross_attention_dim)])
    unet.encoder_hid_proj.to(unet.device)

    return unet