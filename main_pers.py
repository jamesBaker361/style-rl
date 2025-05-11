import os
import argparse
from experiment_helpers.gpu_details import print_details
from experiment_helpers.better_vit_model import BetterViTModel
from experiment_helpers.better_ddpo_trainer import BetterDDPOTrainer
from experiment_helpers.unsafe_stable_diffusion_pipeline import UnsafeStableDiffusionPipeline
from pipelines import CompatibleLatentConsistencyModelPipeline
from datasets import load_dataset
import torchvision.transforms as transforms

import torch
from accelerate import Accelerator
import time
from diffusers.models.embeddings import IPAdapterFullImageProjection
from extractor import ViTExtractor
import torch.nn.functional as F
from PIL import Image
import random
from transformers import AutoImageProcessor, Dinov2Model, BaseImageProcessorFast, SiglipModel
from worse_peft import apply_lora
import wandb
import numpy as np
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from transformers.models.siglip.image_processing_siglip_fast import SiglipImageProcessorFast
from transformers.models.siglip.processing_siglip import SiglipProcessor
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil

seed=1234
random.seed(seed)                      # Python
np.random.seed(seed)                   # NumPy
torch.manual_seed(seed)                # PyTorch (CPU)
try:
    torch.cuda.manual_seed(seed)           # PyTorch (GPU)
except:
    pass



parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--data_dir",type=str,default="data_dir")
parser.add_argument("--save_data_npz",action="store_true")
parser.add_argument("--load_data_npz",action="store_true")
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--training_type",help="denoise or reward",default="denoise",type=str)
parser.add_argument("--prediction_type",type=str,default="epsilon",help="epsilon or v_prediction")
parser.add_argument("--train_split",type=float,default=0.96)
parser.add_argument("--validation_interval",type=int,default=20)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--cross_attention_dim",type=int,default=1024)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--dino_pooling_stride",default=4,type=int)

import torch
import torch.nn.functional as F

def make_batches_same_size(tensor_list, batch_size):
    """
    Splits a list of equally‑sized tensors into batches.

    Args:
        tensor_list (List[torch.Tensor]): List of tensors, all with identical shape.
        batch_size (int): Desired batch size.

    Returns:
        List[torch.Tensor]: List of batched tensors, each of shape (batch_size, *tensor_shape).
    """
    batches = []
    for i in range(0, len(tensor_list), batch_size):
        batch = tensor_list[i:i + batch_size]
        batched = torch.stack(batch, dim=0)
        batches.append(batched)
    return batches

def split_list_by_ratio(lst, ratios=(0.8, 0.1, 0.1)):
    #assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    n = len(lst)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return lst[:i1], lst[i1:i2], lst[i2:]




def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    with accelerator.autocast():
        try:
            raw_data=load_dataset(args.dataset,split="train")
        except OSError:
            raw_data=load_dataset(args.dataset,split="train",force_download=True)

        os.makedirs(args.data_dir,exist_ok=True)

        embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

        
        
        embedding_list=[]
        text_list=[]
        image_list=[]
        shuffled_row_list=[row for row in raw_data]
        random.shuffle(shuffled_row_list)
        with torch.no_grad():
            for i,row in enumerate(raw_data):
                if i==args.limit:
                    break
                before_objects=find_cuda_objects()
                image=row["image"]
                image_list.append(embedding_util.transform_image(image).to("cpu"))
                text=row["text"]
                if type(text)==list:
                    text=text[0]
                
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0)
                #print(embedding.size())
                embedding.to("cpu")
                embedding_list.append(embedding)
                accelerator.free_memory()
                torch.cuda.empty_cache()
                
                text_list.append(text)
                #print(get_gpu_memory_usage())
                #print("gpu objects:",len(find_cuda_objects()))
                after_objects=find_cuda_objects()
                delete_unique_objects(after_objects,before_objects)
                #print("grads",len(find_cuda_tensors_with_grads()))


        def loss_fn(img_tensor_batch:torch.Tensor, src_embedding_batch:torch.Tensor)->torch.Tensor:
            pred_embedding_batch=embedding_util.embed_img_tensor(img_tensor_batch)
            return F.mse_loss(pred_embedding_batch,src_embedding_batch)
        
        fake_image=torch.rand((1,3,args.image_size,args.image_size))
        fake_embedding=embedding_util.embed_img_tensor(fake_image)
        embedding_dim=fake_embedding.size()[-1]

        print("embedding dim",embedding_dim)

        if args.pipeline=="lcm":
            pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        vae=pipeline.vae
        unet=pipeline.unet
        text_encoder=pipeline.text_encoder
        scheduler=pipeline.scheduler
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        '''vae.to(device,torch_dtype)
        unet.to(device,torch_dtype)
        text_encoder.to(device,torch_dtype)
        scheduler.to(device,torch_dtype)'''
        #pipeline.requires_grad_(False)
        for component in [vae,unet,text_encoder]:
            component.to(device,torch_dtype)
            component.requires_grad_(False)
        
        replace_ip_attn(unet,args.cross_attention_dim,embedding_dim)
        attn_layer_list=[p for (name,p ) in get_modules_of_types(unet,IPAdapterAttnProcessor2_0)]
        attn_layer_list.append( unet.encoder_hid_proj)
        print("len attn_layers",len(attn_layer_list))
        for layer in attn_layer_list:
            layer.requires_grad_(True)


        print("before ",pipeline.unet.config.sample_size)
        pipeline.unet.config.sample_size=args.image_size // pipeline.vae_scale_factor
        print("after", pipeline.unet.config.sample_size)
        
        ratios=(args.train_split,(1.0-args.train_split)/2.0,(1.0-args.train_split)/2.0)
        print('train/test/val',ratios)
        batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
        batched_embedding_list,test_batched_embedding_list,val_batched_embedding_list=split_list_by_ratio(batched_embedding_list,ratios)

        
        batched_image_list= image_list #make_batches_same_size(image_list,args.batch_size)
        batched_text_list= text_list #[text_list[i:i + args.batch_size] for i in range(0, len(text_list), args.batch_size)]

        
        batched_image_list,test_batched_image_list,val_batched_image_list=split_list_by_ratio(batched_image_list,ratios)
        batched_text_list,test_batched_text_list,val_batched_text_list=split_list_by_ratio(batched_text_list,ratios)

        

        params=[p for p in pipeline.unet.parameters() if p.requires_grad]

        print("trainable params: ",len(params))

        optimizer=torch.optim.AdamW(params)

        vae,unet,text_encoder,scheduler,optimizer,pipeline=accelerator.prepare(vae,unet,text_encoder,scheduler,optimizer,pipeline)

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def logging(batched_text_list, batched_embedding_list, batched_image_list,pipeline,baseline:bool=False,auto_log:bool=True):
            metrics={}
            difference_list=[]
            embedding_difference_list=[]
            text_alignment_list=[]
            for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(batched_text_list, batched_embedding_list, batched_image_list)):
                image_embeds=embeds_batch #.unsqueeze(0)
                prompt=text_batch
                if random.random() <args.uncaptioned_frac:
                    prompt=" "
                image_batch=torch.clamp(image_batch, 0, 1)
                if baseline:
                    ip_adapter_image=F_v2.resize(image_batch, (224,224)).unsqueeze(0)
                    image=pipeline(prompt,ip_adapter_image=ip_adapter_image,output_type="pt",height=args.image_size,width=args.image_size).images[0]
                else:
                    image=pipeline(prompt,ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
                image_batch=F_v2.resize(image_batch, (args.image_size,args.image_size))
                print("img vs real img",image.size(),image_batch.size())
                #image_embeds.to("cpu")
                image_batch=image_batch.to(image.device)

                difference_list.append(F.mse_loss(image,image_batch).cpu().detach().item())


                embedding_real=embedding_util.embed_img_tensor(image_batch)
                embedding_fake=embedding_util.embed_img_tensor(image)
                embedding_difference_list.append(F.mse_loss(embedding_real,embedding_fake).cpu().detach().item())
                
                
                do_denormalize= [True] * image.shape[0]
                pil_image=pipeline.image_processor.postprocess(image.unsqueeze(0),"pil",do_denormalize)[0]
                if prompt!=" ":
                    inputs = clip_processor(
                        text=[prompt], images=pil_image, return_tensors="pt", padding=True
                    )

                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    text_alignment_list.append(logits_per_image.cpu().detach().item())

                metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(pil_image)
            metrics["difference"]=np.mean(difference_list)
            metrics["embedding_difference"]=np.mean(embedding_difference_list)
            metrics["text_alignment"]=np.mean(text_alignment_list)
            if auto_log:
                accelerator.log(metrics)
            return metrics

        training_start=time.time()
        for e in range(1, args.epochs+1):
            before_objects=find_cuda_objects()
            start=time.time()
            loss_buffer=[]
            for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(batched_text_list, batched_embedding_list, batched_image_list)):
                print(b,len(text_batch), 'embeds',embeds_batch.size(), "img", image_batch.size())
                embeds_batch.to(device,torch_dtype)
                image_embeds=embeds_batch #.unsqueeze(1)
                print('image_embeds',image_embeds.requires_grad,image_embeds.size())
                prompt=text_batch
                if args.epochs >1 and  random.random() <args.uncaptioned_frac:
                    prompt=" "
                if args.training_type=="denoise":
                    with accelerator.accumulate(params):
                        # Convert images to latent space
                        image_batch=image_batch.to(device,torch_dtype).unsqueeze(0)
                        latents = vae.encode(image_batch).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        #print('latents',latents.requires_grad,latents.size())

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)

                            # Get the text embedding for conditioning
                        prompt_embeds, _ = pipeline.encode_prompt(
                                prompt,
                                accelerator.device,
                                1,
                                pipeline.do_classifier_free_guidance,
                                negative_prompt=None,
                                prompt_embeds=None,
                                negative_prompt_embeds=None,
                                #lora_scale=lora_scale,
                        )

                        
                        encoder_hidden_states = prompt_embeds
                        #print("encoede hiiden states",encoder_hidden_states.requires_grad)
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch_size,), device=latents.device)
                        #timesteps = timesteps.long()

                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                        # Get the target for loss depending on the prediction type
                        if args.prediction_type is not None:
                            # set prediction_type of scheduler if defined
                            scheduler.register_to_config(prediction_type=args.prediction_type)

                        if scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif scheduler.config.prediction_type == "v_prediction":
                            target = scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
                        
                        added_cond_kwargs={"image_embeds":image_embeds}

                        # Predict the noise residual and compute loss
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]

                        print('model_pred',model_pred.requires_grad)

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()
                elif args.training_type=="reward":
                    with accelerator.accumulate(params):
                        images=pipeline.call_with_grad(prompt=prompt, num_inference_steps=args.num_inference_steps, ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
                        predicted=embedding_util.embed_img_tensor(images)
                        loss=loss_fn(images,predicted)
                        #loss=(loss-np.mean(loss_buffer))/np.std(loss_buffer)
                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()

                loss_buffer.append(loss.cpu().detach().item())
                if torch.cuda.is_available():
                    before_memory=get_gpu_memory_usage()["allocated_mb"]
                after_objects=find_cuda_objects()
                len_after=len(after_objects)
                delete_unique_objects(before_objects,after_objects)
                after_after_objects=find_cuda_objects()
                print("deleted ",len_after-len(after_after_objects))
                if torch.cuda.is_available():
                    after_memory=get_gpu_memory_usage()["allocated_mb"]
                    print(f"freed {before_memory-after_memory} mb")

            end=time.time()
            elapsed=end-start
            print(f"\t epoch {e} elapsed {end-start}")
            accelerator.log({
                "loss_mean":np.mean(loss_buffer),
                "loss_std":np.std(loss_buffer),
                "elapsed":elapsed
            })
            accelerator.free_memory()
            if e%args.validation_interval==0:
                before_objects=find_cuda_objects()
                with torch.no_grad():

                    start=time.time()
                    logging(val_batched_text_list,val_batched_embedding_list,val_batched_image_list,pipeline)
                    end=time.time()
                    print(f"\t validation epoch {e} elapsed {end-start}")
                after_objects=find_cuda_objects()
                delete_unique_objects(after_objects,before_objects)
        training_end=time.time()
        print(f"total trainign time = {training_end-training_start}")
        accelerator.free_memory()
        metrics=logging(test_batched_text_list,test_batched_embedding_list,test_batched_image_list,pipeline,auto_log=False)
        new_metrics={}
        for k,v in metrics.items():
            new_metrics["test_"+k]=v
        accelerator.log(new_metrics)

        if args.pipeline=="lcm":
            baseline_pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        baseline_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        baseline_metrics=logging(test_batched_text_list,test_batched_embedding_list,test_batched_image_list,baseline_pipeline,baseline=True)
        new_metrics={}
        for k,v in baseline_metrics.items():
            new_metrics["baseline_"+k]=v
        accelerator.log(new_metrics)

        

                

    


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