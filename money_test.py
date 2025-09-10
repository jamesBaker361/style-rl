from ipattn import *
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
    IPAdapterXFormersAttnProcessor,
    JointAttnProcessor2_0,
    SD3IPAdapterJointAttnProcessor2_0,
)
from diffusers.models.attention_processor import  IPAdapterAttnProcessor2_0,Attention

pipe = StableDiffusionPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    )

    # Load IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

unet=pipe.unet

count=0
for attn_name, attn_processor in unet.attn_processors.items():
    if isinstance(
        attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor)):
        count+=1

print(count)

attn_list=get_modules_of_types(pipe.unet,Attention)
for [name,_] in attn_list:
    print(name)
    if name in unet.attn_processors:
        print("match !")
