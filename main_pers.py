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
from transformers import AutoImageProcessor, Dinov2Model, BaseImageProcessorFast
from worse_peft import apply_lora
import wandb
import numpy as np
import random

parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--embedding",type=str,default="dino")
parser.add_argument("--reward_embedding",type=str,default="dino")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--data_dir",type=str,default="data_dir")
parser.add_argument("--save_data_npz",action="store_true")
parser.add_argument("--load_data_npz",action="store_true")
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--training_type",help="denoise or reward",default="denoise")
parser.add_argument("--train_unet",action="store_true")
parser.add_argument("--prediction_type",type=str,default="epsilon")
parser.add_argument("--train_split",type=float,default=0.96)
parser.add_argument("--validation_interval",type=int,default=20)
parser.add_argument("--buffer_size",type=int,default=0)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)

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
        
        raw_data=load_dataset(args.dataset,split="train")

        os.makedirs(args.data_dir,exist_ok=True)

        if args.embedding=="dino":
            dino_vit_extractor=ViTExtractor("dino_vits8",device=accelerator.device)
            dino_vit_extractor.model.eval()
            dino_vit_extractor.model.requires_grad_(False)
        elif args.embedding=="ssl":
            processor = BaseImageProcessorFast.from_pretrained('facebook/webssl-dino1b-full2b-224')
            model = Dinov2Model.from_pretrained('facebook/webssl-dino1b-full2b-224')
            model.to(device,torch_dtype)

        def embed_img_tensor(img_tensor:torch.Tensor)->torch.Tensor:
            img_tensor=img_tensor.to(device,torch_dtype)
            if args.embedding=="dino":
                if len(img_tensor.size())==3:
                    img_tensor=img_tensor.unsqueeze(0)
                img_tensor=F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                #dino_vit_prepocessed=dino_vit_extractor.preprocess_pil(content_image.resize((args.image_size,args.image_size))).to(dtype=torch_dtype,device=accelerator.device)
                dino_vit_features=dino_vit_extractor.extract_descriptors(img_tensor,facet=args.facet)
                batch_size=img_tensor.size()[0]
                print('dino_vit_features.size()',dino_vit_features.size())
                embedding=dino_vit_features.view(batch_size,-1)
            elif args.embedding=="ssl":
                #print("before ",type(img_tensor),img_tensor.size())
                p_inputs=processor(img_tensor,return_tensors="pt")
                #print(p_inputs)
                outputs = model(**p_inputs)
                cls_features = outputs.last_hidden_state[:, 0]  # CLS token features
                #print("cls featurs size",cls_features.size())
                embedding=cls_features
                

            return embedding
        
        def transform_image(pil_image:Image.Image):
            if args.embedding=="dino":
                t=transforms.Compose(
                    [transforms.ToTensor(),transforms.Normalize(dino_vit_extractor.mean,dino_vit_extractor.std)]
                )
            elif args.embedding=="ssl":
                t=transforms.Compose(
                    [transforms.ToTensor()]
                )
            return t(pil_image)
        
        embedding_list=[]
        text_list=[]
        image_list=[]
        shuffled_row_list=[row for row in raw_data]
        random.shuffle(shuffled_row_list)
        for row in raw_data:
            image=row["image"]
            image_list.append(transform_image(image))
            text=row["text"]
            if type(text)==list:
                text=text[0]
            embedding_list.append(embed_img_tensor(transform_image(image))[0]).to("cpu")
            text_list.append(text)

        def loss_fn(img_tensor_batch:torch.Tensor, src_embedding_batch:torch.Tensor)->torch.Tensor:
            pred_embedding_batch=embed_img_tensor(img_tensor_batch)
            return F.mse_loss(pred_embedding_batch,src_embedding_batch)
        
        fake_image=torch.rand((1,3,args.image_size,args.image_size))
        fake_embedding=embed_img_tensor(fake_image)
        embedding_dim=fake_embedding.size()[-1]

        print("embedding dim",embedding_dim)

        if args.pipeline=="lcm":
            pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
            vae=pipeline.vae
            unet=pipeline.unet
            text_encoder=pipeline.text_encoder
            scheduler=pipeline.scheduler
            '''vae.to(device,torch_dtype)
            unet.to(device,torch_dtype)
            text_encoder.to(device,torch_dtype)
            scheduler.to(device,torch_dtype)'''
            #pipeline.requires_grad_(False)
            for component in [vae,unet,text_encoder]:
                component.to(device,torch_dtype)
                component.requires_grad_(False)

        print("before ",pipeline.unet.config.sample_size)
        pipeline.unet.config.sample_size=args.image_size // pipeline.vae_scale_factor
        print("after", pipeline.unet.config.sample_size)
        
        ratios=(args.train_split,(1-args.train_split)//2,(1-args.train_split)//2)
        print(ratios)
        batched_embedding_list=make_batches_same_size(embedding_list,args.batch_size)
        batched_embedding_list,test_batched_embedding_list,val_batched_embedding_list=split_list_by_ratio(batched_embedding_list,ratios)

        text_embedding_list=[]
        for t in text_list:
            text_embeds, _ = pipeline.encode_prompt(
                    text,
                    accelerator.device,
                    1,
                    pipeline.do_classifier_free_guidance,
                    negative_prompt=None,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                )
            print('textembeds.size',text_embeds.size())

            text_embedding_list.append(text_embeds[0])

        batched_text_embedding_list=make_batches_same_size(text_embedding_list,args.batch_size)
        batched_image_list=make_batches_same_size(image_list,args.batch_size)
        batched_text_list= [text_list[i:i + args.batch_size] for i in range(0, len(text_list), args.batch_size)]

        
        batched_image_list,test_batched_image_list,val_batched_image_list=split_list_by_ratio(batched_image_list,ratios)
        batched_text_list,test_batched_text_list,val_batched_text_list=split_list_by_ratio(batched_text_list,ratios)

        #the output of the embeddign thing can be passed as ip_adapter_image_embeds or the image itself can be passed as     ip_adapter_image to the pipeline
        #multiple projection layers for different layers..?


        cross_attention_dim=unet.config.cross_attention_dim
        projection_layer=IPAdapterFullImageProjection(embedding_dim,cross_attention_dim)
        projection_layer.to(device,torch_dtype)
        projection_layer.ff.to(device,torch_dtype)
        projection_layer.requires_grad_(True)

        params=[p for p in projection_layer.parameters()]

        if args.train_unet:
            apply_lora(pipeline.unet,[0,1,2,3],[0,1,2,3],True)
            params+=[p for p in pipeline.unet.params() if p.requires_grad]

        print("trainable params: ",len(params))

        optimizer=torch.optim.AdamW(params)

        vae,unet,text_encoder,scheduler,optimizer,pipeline,projection_layer=accelerator.prepare(vae,unet,text_encoder,scheduler,optimizer,pipeline,projection_layer)
        '''if args.training_type=="reward":
            loss_buffer=[]
            for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(batched_text_list, batched_embedding_list, batched_image_list)):
                if b==args.buffer_size:
                    break
                text=text_batch[0]
                image=pipeline(text,output_type="pt").images[0]
                loss=loss_fn(image,embeds_batch[0])
                loss_buffer.append(loss.cpu().detach().item())'''

        training_start=time.time()
        for e in range(1, args.epochs+1):
            start=time.time()
            loss_buffer=[]
            for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(batched_text_list, batched_embedding_list, batched_image_list)):
                print(b,len(text_batch), 'embeds',embeds_batch.size(), "img", image_batch.size())
                embeds_batch.to(device,torch_dtype)
                image_embeds=projection_layer(embeds_batch)
                image_embeds=image_embeds.unsqueeze(1)
                #print(image_embeds.size())
                prompt=text_batch[0]
                if args.epochs >1 and  random.random() <args.uncaptioned_frac:
                    prompt=" "
                if args.training_type=="denoise":
                    with accelerator.accumulate(params):
                        # Convert images to latent space
                        latents = vae.encode(image_batch).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

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
                        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (args.batch_size,), device=latents.device)
                        #timesteps = timesteps.long()

                        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                        # Get the target for loss depending on the prediction type
                        if args.prediction_type is not None:
                            # set prediction_type of scheduler if defined
                            scheduler.register_to_config(prediction_type=args.prediction_type)

                        if scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif scheduler.config.prediction_type == "v_prediction" or scheduler.config.prediction_type =="velocity":
                            target = scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
                        
                        added_cond_kwargs={"image_embeds":image_embeds}

                        # Predict the noise residual and compute loss
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]

                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()
                elif args.training_type=="reward":
                    with accelerator.accumulate():
                        images=pipeline.call_with_grad(prompt=prompt, ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
                        predicted=embed_img_tensor(images)
                        loss=loss_fn(images,predicted)
                        #loss=(loss-np.mean(loss_buffer))/np.std(loss_buffer)
                        accelerator.backward(loss)

                        optimizer.step()
                        optimizer.zero_grad()

                loss_buffer.append(loss.cpu().detach().item())
            end=time.time()
            elapsed=end-start
            print(f"\t epoch {e} elapsed {end-start}")
            accelerator.log({
                "loss_mean":np.mean(loss_buffer),
                "loss_std":np.std(loss_buffer)
            })
            accelerator.free_memory()
            if e%args.validation_interval==0:
                start=time.time()
                metrics={}
                difference_list=[]
                embedding_difference_list=[]
                start=time.time()
                for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(val_batched_text_list, val_batched_embedding_list, val_batched_image_list)):
                    image_embeds=projection_layer(embeds_batch)
                    image_embeds=image_embeds.unsqueeze(1)
                    prompt=" "
                    image=pipeline(prompt,ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
                    difference_list.append(F.mse_loss(image,image_batch[0]).cpu().detach().item())
                    pil_image=pipeline.image_processor.postprocess(image,"pil",[True])
                    metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(pil_image)
                metrics["difference"]=np.mean(difference_list)
                accelerator.log(metrics)
                end=time.time()
                print(f"\t validation epoch {e} elapsed {end-start}")
        training_end=time.time()
        print(f"total trainign time = {training_end-training_start}")
        accelerator.free_memory()
        difference_list=[]
        metrics={}
        for b,(text_batch, embeds_batch,image_batch) in enumerate(zip(test_batched_text_list, test_batched_embedding_list, test_batched_image_list)):
            image_embeds=projection_layer(embeds_batch)
            image_embeds=image_embeds.unsqueeze(1)
            prompt=" "
            image=pipeline(prompt,ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
            difference_list.append(F.mse_loss(image,image_batch[0]).cpu().detach().item())
            pil_image=pipeline.image_processor.postprocess(image,"pil",[True])
            metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(pil_image)
        metrics["difference"]=np.mean(difference_list)
        accelerator.log(metrics)
        

                

    


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