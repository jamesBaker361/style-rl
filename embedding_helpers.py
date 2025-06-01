import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from extractor import ViTExtractor
from extractor import ViTExtractor
import torch.nn.functional as F
from PIL import Image
import random
from custom_processor import CustomProcessor
from transformers import (
    AutoImageProcessor,
    Dinov2Model,                   
    SiglipModel,           
    CLIPProcessor,              
    CLIPVisionModel,             
    CLIPVisionModelWithProjection ,
    BaseImageProcessorFast
)

from transformers.models.siglip.processing_siglip import SiglipProcessor

def inverse_tokenize(x):
            # x: (B, N, embed_dim)
            B, N, C = x.shape
            H = W = 224 // 16  # assuming square image/patch
            x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, embed_dim, H/patch, W/patch)
            return x


class EmbeddingUtil():
    def __init__(self,device,torch_dtype,
                     embedding:str,
                     facet:str,
                     dino_pooling_stride:int):
        self.device=device
        self.torch_dtype=torch_dtype
        self.embedding=embedding
        self.facet=facet
        self.dino_pooling_stride=dino_pooling_stride
        if embedding=="dino":
            self.dino_vit_extractor=ViTExtractor("dino_vits16",device=device,stride=16)
            self.dino_vit_extractor.model=self.dino_vit_extractor.model.to(device,torch_dtype)
            self.dino_vit_extractor.model.eval()
            self.dino_vit_extractor.model.requires_grad_(False)
        elif embedding=="ssl":
            self.ssl_processor = BaseImageProcessorFast.from_pretrained('facebook/webssl-dino1b-full2b-224')
            self.ssl_model = Dinov2Model.from_pretrained('facebook/webssl-dino1b-full2b-224')
            self.ssl_model.to(device,torch_dtype)
            self.ssl_model.requires_grad_(False)
        elif embedding=="siglip2":
            self.siglip_model = SiglipModel.from_pretrained("google/siglip2-base-patch16-224")
            self.siglip_processor = CustomProcessor()
            #SiglipProcessor.image_processor=SiglipImageProcessorFast.from_pretrained("google/siglip2-base-patch16-224",do_convert_rgb=False,device=torch.cuda.get_device_name(device))
            self.siglip_model.to(device,torch_dtype)
            self.siglip_model.requires_grad_(False)
        elif embedding=="clip":
            self.clip_model=CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor=CustomProcessor(image_mean=[0.48145466,0.4578275,0.40821073],
                                                image_std=[0.26862954,0.26130258,0.27577711])
            self.clip_model.to(device,torch_dtype)
            self.clip_model.requires_grad_(False)

    def embed_img_tensor(self,img_tensor:torch.Tensor,
                        )->torch.Tensor:
        img_tensor=img_tensor.to(self.device,self.torch_dtype)
        if self.embedding=="dino":
            if len(img_tensor.size())==3:
                img_tensor=img_tensor.unsqueeze(0)
            img_tensor=F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            #dino_vit_prepocessed=dino_vit_extractor.preprocess_pil(content_image.resize((args.image_size,args.image_size))).to(dtype=torch_dtype,device=accelerator.device)
            img_tensor=(img_tensor+1)/2 #convert from [-1,1] to 0,1
            norm=transforms.Normalize(self.dino_vit_extractor.mean,self.dino_vit_extractor.std)
            img_tensor=norm(img_tensor) #normalize using imagenet statistics
            dino_vit_features=self.dino_vit_extractor.extract_descriptors(img_tensor,facet=self.facet)
            batch_size=img_tensor.size()[0]
            #print('dino_vit_features.size()',dino_vit_features.size())
            dino_vit_features=inverse_tokenize(dino_vit_features)
            dino_vit_features=F.max_pool2d(dino_vit_features, kernel_size=self.dino_pooling_stride, stride=self.dino_pooling_stride)
            embedding=dino_vit_features.reshape(batch_size,-1)
        elif self.embedding=="ssl":
            #print("before ",type(img_tensor),img_tensor.size())
            p_inputs=self.ssl_processor(img_tensor,return_tensors="pt")
            #print(p_inputs)
            outputs = self.ssl_model(**p_inputs)
            cls_features = outputs.last_hidden_state[:, 0]  # CLS token features
            #print("cls featurs size",cls_features.size())
            embedding=cls_features
        elif self.embedding=="siglip2":
            #print("img",img_tensor.device)
            inputs = self.siglip_processor(images=img_tensor)
            '''for key in ['input_ids','pixel_values']:
                inputs[key]=inputs[key].to(self.device)'''
            if len(inputs["pixel_values"].size())==3:
                 inputs["pixel_values"]=inputs["pixel_values"].unsqueeze(0)
            outputs = self.siglip_model.vision_model(pixel_values=inputs["pixel_values"],output_attentions=False,output_hidden_states=False)
            embedding=outputs.pooler_output
        elif self.embedding=="clip":
            inputs=self.clip_processor(images=img_tensor)
            #inputs["pixel_values"]=inputs["pixel_values"].to(self.device)
            if len(inputs["pixel_values"].size())==3:
                inputs["pixel_values"]=inputs["pixel_values"].unsqueeze(0)
            outputs=self.clip_model.vision_model(pixel_values=inputs["pixel_values"],output_attentions=False,output_hidden_states=False)
            embedding=outputs.pooler_output
            
        return embedding

    def transform_image(self,pil_image:Image.Image):
        if self.embedding=="dino":
            t=transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize(self.dino_vit_extractor.mean,self.dino_vit_extractor.std)]
            )
        elif self.embedding=="ssl" or self.embedding=="siglip2" or self.embedding=="clip":
            t=transforms.Compose(
                [transforms.ToTensor()]
            )
        return t(pil_image)