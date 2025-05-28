from diffusers.models.embeddings import IPAdapterFullImageProjection,MultiIPAdapterImageProjection,ImageProjection
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from diffusers import UNet2DConditionModel
import torch
from typing import List

class MultiIPAdapterImageProjectionWithVisualProjection(torch.nn.Module):
    def __init__(self, multi_ip_adapter:MultiIPAdapterImageProjection,
                 embedding_dim:int, #embedding dim from the embedding model
                 intermediate_embedding_dim:int, #embedding dim that is NOT the cross attention dim or the result of the embedding model
                 device,
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.multi_ip_adapter=multi_ip_adapter.to(device)
        self.visual_projection=torch.nn.Linear(embedding_dim,intermediate_embedding_dim,bias=False).to(device)

    def forward(self,  image_embeds: List[torch.Tensor]):
        image_embeds=[self.visual_projection(image) for image in image_embeds]
        return self.multi_ip_adapter(image_embeds)


def get_modules_of_types(model, target_classes):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, target_classes)]

def replace_ip_attn(unet:UNet2DConditionModel
                    ,embedding_dim:int,
                    intermediate_embedding_dim:int,
                    cross_attention_dim:int
                    ,num_image_text_embeds:int):
    layers=get_modules_of_types(unet,IPAdapterAttnProcessor2_0)
    for (name,module) in layers:
        out_features=module.to_k_ip[0].out_features
        new_k_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features,bias=False)])
        new_k_ip.to(unet.device)
        setattr(module, "to_k_ip",new_k_ip)
        new_v_ip=torch.nn.ModuleList([torch.nn.Linear(cross_attention_dim,out_features,bias=False)])
        new_v_ip.to(unet.device)
        setattr(module, "to_v_ip",new_v_ip)

    multi_ip_adapter=MultiIPAdapterImageProjection([ImageProjection(embedding_dim,cross_attention_dim,num_image_text_embeds)]).to(device=unet.device)
    #unet.add_module("encoder_hid_proj",multi_ip_adapter)
    #unet.encoder_hid_proj=multi_ip_adapter
    multi_ip_projection=MultiIPAdapterImageProjectionWithVisualProjection(multi_ip_adapter,embedding_dim,intermediate_embedding_dim)
    unet.encoder_hid_proj= multi_ip_projection
    unet.add_module("encoder_hid_proj",multi_ip_projection)
    #unet.encoder_hid_proj.to(unet.device)


    return unet