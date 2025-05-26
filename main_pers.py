import os
import argparse
from experiment_helpers.gpu_details import print_details
from pipelines import CompatibleLatentConsistencyModelPipeline
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
from accelerate import Accelerator
import time
import torch.nn.functional as F
from PIL import Image
import random
from worse_peft import apply_lora
import wandb
import numpy as np
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil
from data_helpers import CustomTripleDataset
from custom_vae import public_encode
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

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
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--epochs",type=int,default=10)
parser.add_argument("--training_type",help="denoise or reward",default="denoise",type=str)
parser.add_argument("--prediction_type",type=str,default="epsilon",help="epsilon or v_prediction")
parser.add_argument("--train_split",type=float,default=0.5)
parser.add_argument("--validation_interval",type=int,default=20)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--intermediate_embedding_dim",type=int,default=1024)
parser.add_argument("--cross_attention_dim",type=int,default=768)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--num_inference_steps",type=int,default=4)
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--num_image_text_embeds",type=int,default=4)
parser.add_argument("--deepspeed",action="store_true",help="whether to use deepspeed")
parser.add_argument("--fsdp",action="store_true",help=" whether to use fsdp training")
parser.add_argument("--vanilla",action="store_true",help="no distribution")

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
    if args.deepspeed:
        accelerator=Accelerator(log_with="wandb")
        print("using deepspeed")
    else:
        accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]



    #with accelerator.autocast():
    try:
        raw_data=load_dataset(args.dataset,split="train")
    except OSError:
        raw_data=load_dataset(args.dataset,split="train",download_mode="force_redownload")

    os.makedirs(args.data_dir,exist_ok=True)

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)

    

    if args.pipeline=="lcm":
        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
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
    embedding_list=[]
    text_list=[]
    image_list=[]
    posterior_list=[]
    shuffled_row_list=[row for row in raw_data]
    random.shuffle(shuffled_row_list)
    composition=transforms.Compose([
            transforms.Resize((args.image_size,args.image_size)),
             transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
        ])
    with torch.no_grad():
        for i,row in enumerate(raw_data):
            if i==args.limit:
                break
            before_objects=find_cuda_objects()
            image=row["image"]
            image=composition(image)
            posterior=public_encode(vae,image.unsqueeze(0)).squeeze(0)
            posterior_list.append(posterior)
            image_list.append(image)
            text=row["text"]
            if type(text)==list:
                text=text[0]
            
            if "embedding" in row:
                #print(row["embedding"])
                np_embedding=np.array(row["embedding"])
                #print("np_embedding",np_embedding.shape)
                embedding=torch.from_numpy(np_embedding)
                #print("embedding",embedding.size())
                #real_embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0)
                #print("real embedding",real_embedding.size())
            else:
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image))
            #print(embedding.size())
            embedding.to("cpu")
            embedding_list.append(embedding)
            accelerator.free_memory()
            torch.cuda.empty_cache()
            
            
            text, _ = pipeline.encode_prompt(
                                    text,
                                    "cpu", #accelerator.device,
                                    1,
                                    pipeline.do_classifier_free_guidance,
                                    negative_prompt=None,
                                    prompt_embeds=None,
                                    negative_prompt_embeds=None,
                                    #lora_scale=lora_scale,
                            )
            text=text.squeeze(0)
            if i ==1:
                print("text size",text.size(),"embedding size",embedding.size(),"img size",image.size(),"latent size",posterior.size())
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

    for component in [vae,text_encoder]:
        component.requires_grad_(False)
        component.to("cpu")
        #unet=unet.to(device,torch_dtype)
    
    replace_ip_attn(unet,
                    embedding_dim,
                    args.intermediate_embedding_dim,
                    args.cross_attention_dim,
                    args.num_image_text_embeds)
    #print("image projection",unet.encoder_hid_proj.multi_ip_adapter.image_projection_layers[0])
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
    #batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
    embedding_list,test_embedding_list,val_embedding_list=split_list_by_ratio(embedding_list,ratios)

    
    #image_list= image_list #make_batches_same_size(image_list,args.batch_size)
    #text_list= text_list #[text_list[i:i + args.batch_size] for i in range(0, len(text_list), args.batch_size)]

    
    image_list,test_image_list,val_image_list=split_list_by_ratio(image_list,ratios)
    text_list,test_text_list,val_text_list=split_list_by_ratio(text_list,ratios)

    posterior_list,test_posterior_list,val_posterior_list=split_list_by_ratio(posterior_list,ratios)

    train_dataset=CustomTripleDataset(image_list,embedding_list,text_list,posterior_list)
    val_dataset=CustomTripleDataset(val_image_list,val_embedding_list,val_text_list,val_posterior_list)
    test_dataset=CustomTripleDataset(test_image_list,test_embedding_list,test_text_list,test_posterior_list)

    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size)
    test_loader=DataLoader(test_dataset,args.batch_size)

    params=[p for p in pipeline.unet.parameters() if p.requires_grad]

    print("trainable params: ",len(params))

    optimizer=torch.optim.AdamW(params)

    if args.vanilla:
        unet=unet.to(device,torch_dtype)


    unet,scheduler,optimizer,train_loader,test_loader,val_loader=accelerator.prepare(unet,scheduler,optimizer,train_loader,test_loader,val_loader)

    pipeline.unet=unet

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    fid = FrechetInceptionDistance(feature=2048,normalize=True)

    def logging(data_loader,pipeline,baseline:bool=False,auto_log:bool=True):
        metrics={}
        difference_list=[]
        embedding_difference_list=[]
        text_alignment_list=[]
        fid_list=[]

        for b,batch in enumerate(data_loader):
            '''if args.vanilla:
                for k,v in batch.items():
                    batch[k]=v.to(device,torch_dtype)'''
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            if len(image_batch.size())==3:
                image_batch=image_batch.unsqueeze(0)
                text_batch=[text_batch]
                embeds_batch=embeds_batch.unsqueeze(0)
            image_embeds=embeds_batch #.unsqueeze(0)
            
            image_batch=torch.clamp(image_batch, 0, 1)
            if baseline:
                ip_adapter_image=F_v2.resize(image_batch, (224,224))
                image=pipeline(prompt_embeds=text_batch,ip_adapter_image=ip_adapter_image,output_type="pt",height=args.image_size,width=args.image_size).images[0]
            else:
                image=pipeline(prompt_embeds=text_batch,ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
            image_batch=F_v2.resize(image_batch, (args.image_size,args.image_size))
            print("img vs real img",image.size(),image_batch.size())
            #image_embeds.to("cpu")
            image_batch=image_batch.to(image.device)

            difference_list.append(F.mse_loss(image,image_batch).cpu().detach().item())


            embedding_real=embedding_util.embed_img_tensor(image_batch)
            embedding_fake=embedding_util.embed_img_tensor(image)
            embedding_difference_list.append(F.mse_loss(embedding_real,embedding_fake).cpu().detach().item())

            fid.update(image_batch.cpu(),real=True)
            fid.update(image.cpu(),real=False)

            fid_list.append(fid.compute().cpu().detach().item())
            
            
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
        metrics["fid"]=np.mean(fid_list)
        if auto_log:
            accelerator.log(metrics)
        return metrics

    training_start=time.time()
    for e in range(1, args.epochs+1):
        before_objects=find_cuda_objects()
        start=time.time()
        loss_buffer=[]
        for b,batch in enumerate(train_loader):
            '''if args.vanilla:
                for k,v in batch.items():
                    batch[k]=v.to(device,torch_dtype)'''
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            posterior_batch=batch["posterior"]
            if len(image_batch.size())==3:
                image_batch=image_batch.unsqueeze(0)
                text_batch=text_batch.unsqueeze(0)
                embeds_batch=embeds_batch.unsqueeze(0)
                posterior_batch=posterior_batch.unsqueeze(0)
            
            if e==1 and b==0:
                print("text size",text_batch.size(),"embedding size",embeds_batch.size(),"img size",image_batch.size(),"latent size",posterior_batch.size())
                print("text size",text_batch.device,"embedding size",embeds_batch.device,"img size",image_batch.device,"latent size",posterior_batch.device)
            image_embeds=embeds_batch #.to(device) #.unsqueeze(1)
            #print('image_embeds',image_embeds.requires_grad,image_embeds.size())
            prompt=text_batch
            if args.epochs >1 and  random.random() <args.uncaptioned_frac:
                prompt=" "
            #print(pipeline.text_encoder)
            if args.training_type=="denoise":
                with accelerator.accumulate(params):
                    # Convert images to latent space
                    #if args.deepspeed:
                    latents = DiagonalGaussianDistribution(posterior_batch).sample()
                    latents = latents * vae.config.scaling_factor

                    #latents=latents.to(device,torch_dtype)

                    #print('latents',latents.requires_grad,latents.size())

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)

                    
                    pipeline.text_encoder.config.max_position_embeddings=pipeline.tokenizer.model_max_length
                    #print(pipeline.text_encoder.config)
                    

                    
                    encoder_hidden_states = text_batch
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
                    
                    added_cond_kwargs={"image_embeds":[image_embeds]}

                    # Predict the noise residual and compute loss
                    
                    #print('unet.encoder_hid_proj.device',unet.encoder_hid_proj.image_projection_layers[0].device)
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]

                    

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()
            elif args.training_type=="reward":
                with accelerator.accumulate(params):
                    latents = DiagonalGaussianDistribution(posterior_batch).sample()
                    images=pipeline.call_with_grad(prompt_embeds=text_batch, 
                                                   #latents=latents, 
                                                   num_inference_steps=args.num_inference_steps, ip_adapter_image_embeds=[image_embeds],output_type="pt").images[0]
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
                logging(val_loader,pipeline)
                end=time.time()
                print(f"\t validation epoch {e} elapsed {end-start}")
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)
    training_end=time.time()
    print(f"total trainign time = {training_end-training_start}")
    accelerator.free_memory()
    metrics=logging(test_loader,pipeline,auto_log=False)
    new_metrics={}
    for k,v in metrics.items():
        new_metrics["test_"+k]=v
    accelerator.log(new_metrics)

    if args.pipeline=="lcm":
        baseline_pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
    baseline_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    baseline_metrics=logging(test_loader,baseline_pipeline,baseline=True)
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