import os
import argparse
from experiment_helpers.gpu_details import print_details
from pipelines import CompatibleLatentConsistencyModelPipeline,CompatibleStableDiffusionPipeline
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from sklearn.decomposition import PCA
import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
from worse_peft import apply_lora
import wandb
import numpy as np
import datasets
import random
from gpu_helpers import *
from adapter_helpers import replace_ip_attn,get_modules_of_types
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance

from transformers import AutoProcessor, CLIPModel
from embedding_helpers import EmbeddingUtil
from data_helpers import CustomDataset
from custom_vae import public_encode
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi
from PIL import Image
from sana_pipelines import CompatibleSanaSprintPipeline, prepare_ip_adapter,compatible_forward_sana_transformer_model
from custom_scheduler import *
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

def concat_images_horizontally(images):
    """
    Concatenate a list of PIL.Image objects horizontally.

    Args:
        images (List[PIL.Image]): List of PIL images.

    Returns:
        PIL.Image: A new image composed of the input images concatenated side-by-side.
    """
    # Resize all images to the same height (optional)
    heights = [img.height for img in images]
    min_height = min(heights)
    resized_images = [
        img if img.height == min_height else img.resize(
            (int(img.width * min_height / img.height), min_height),
            Image.LANCZOS
        ) for img in images
    ]

    # Compute total width and max height
    total_width = sum(img.width for img in resized_images)
    height = min_height

    # Create new blank image
    new_img = Image.new('RGB', (total_width, height))

    # Paste images side by side
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_img


seed=1234
random.seed(seed)                      # Python
np.random.seed(seed)                   # NumPy
torch.manual_seed(seed)                # PyTorch (CPU)
try:
    torch.cuda.manual_seed(seed)           # PyTorch (GPU)
except:
    pass



parser=argparse.ArgumentParser()
parser.add_argument("--dataset",type=str,default="jlbaker361/captioned-images",help="src images to test on")
parser.add_argument("--mixed_precision",type=str,default="no")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--image_size",type=int,default=256)
parser.add_argument("--embedding",type=str,default="dino",help="dino ssl or siglip2")
parser.add_argument("--facet",type=str,default="query",help="dino vit facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']")
parser.add_argument("--pipeline",type=str,default="lcm")
parser.add_argument("--batch_size",type=int,default=1)
parser.add_argument("--train_split",type=float,default=0.5)
parser.add_argument("--uncaptioned_frac",type=float,default=0.75)
parser.add_argument("--intermediate_embedding_dim",type=int,default=1024)
parser.add_argument("--cross_attention_dim",type=int,default=768)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--dino_pooling_stride",default=4,type=int)
parser.add_argument("--num_image_text_embeds",type=int,default=4)
parser.add_argument("--fsdp",action="store_true",help=" whether to use fsdp training")
parser.add_argument("--vanilla",action="store_true",help="no distribution")
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--load",action="store_true",help="whether to load saved version")
parser.add_argument("--load_hf",action="store_true",help="whether to load saved version from hf")
parser.add_argument("--generic_test_prompts",action="store_true")
parser.add_argument("--disable_projection_adapter",action="store_true",help="whether to use projection for ip adapter ")
parser.add_argument("--identity_adapter",action="store_true",help="whether to use identity mapping for IP adapter layers")
parser.add_argument("--deep_to_ip_layers",action="store_true",help="use deeper ip layers")
parser.add_argument("--initial_scale",type=float,default=1.0)
parser.add_argument("--final_scale",type=float,default=1.0)
parser.add_argument("--sigma_data",type=float,default=-0.8)
parser.add_argument("--real_test_prompts",action="store_true")
parser.add_argument("--zeros",action="store_true")
parser.add_argument("--decreasing_scale",action="store_true")
parser.add_argument("--increasing_scale",action="store_true")
parser.add_argument("--fid",action="store_true")
parser.add_argument("--constant_scale",action="store_true")
parser.add_argument("--ip_start",type=float,default=0.0)
parser.add_argument("--ip_end",type=float,default=1.0)
parser.add_argument("--do_classifier_free_guidance",action="store_true")
parser.add_argument("--cfg_embedding",action="store_true")
parser.add_argument("--cfg_weight",type=float,default="3.0")
parser.add_argument("--npz_file",type=str,default="clip_pca_0.95.npz")
parser.add_argument("--pca_project",action="store_true")
parser.add_argument("--baseline_prompt",type=str,default=" ,league of legends style")
parser.add_argument("--classification_data",type=str,default="jlbaker361/league-clip-classification",help="dataset with all of the hyperplane weights")
parser.add_argument("--tagged_data",type=str,default="jlbaker361/league-tagged-clip",help="data to use for normalization stuff")
parser.add_argument("--hyperplane",action="store_true",help="whether to use the hyperplane for alignment with tags")
parser.add_argument("--classifier_type",default="SGD",type=str,help="SGD or SVC classifier")
parser.add_argument("--positive_threshold",type=int,default=5,help="how many positives must exist in the dataset to use the tag")
parser.add_argument("--hyperplane_coefficient",type=float,default=0.1)

import torch
import torch.nn.functional as F

def split_list_by_ratio(lst, ratios=(0.8, 0.1, 0.1)):
    #assert sum(ratios) == 1.0, "Ratios must sum to 1.0"
    n = len(lst)
    i1 = int(n * ratios[0])
    i2 = i1 + int(n * ratios[1])
    return lst[:i1], lst[i1:i2], lst[i2:]




def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    pca_object=PCA(n_components=100)
    with np.load(args.npz_file) as np_dict:
        pca_object.components_=np_dict["components_"]
        pca_object.explained_variance_=np_dict["explained_variance_"]
        pca_object.mean_=np_dict["mean_"]

    






    #with accelerator.autocast():
    try:
        raw_data=load_dataset(args.dataset,split="train")
    except OSError:
        raw_data=load_dataset(args.dataset,split="train",download_mode="force_redownload")
    WEIGHTS_NAME="unet_model.bin"
    CONFIG_NAME="config.json"
    save_dir=os.path.join(os.environ["TORCH_LOCAL_DIR"],args.name)
    save_path=os.path.join(save_dir,WEIGHTS_NAME)
    config_path=os.path.join(save_dir,CONFIG_NAME)
    if accelerator.is_main_process or state.num_processes==1:
        os.makedirs(save_dir,exist_ok=True)

    accelerator.print("\nMODEL-NAME ",args.name.split("/")[-1])
    

    embedding_util=EmbeddingUtil(device,torch_dtype,args.embedding,args.facet,args.dino_pooling_stride)


    adapter_id = "latent-consistency/lcm-lora-sdv1-5"
    if args.pipeline=="lcm":
        pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device)
    elif args.pipeline=="lcm_post_lora":
        pipeline=CompatibleStableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        pipeline.load_lora_weights(adapter_id)
        pipeline.disable_lora()
    elif args.pipeline=="lcm_pre_lora":
        pipeline=CompatibleStableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device)
        
        pipeline.load_lora_weights(adapter_id)
        pipeline.fuse_lora()
    elif args.pipeline=="sana":
        pipeline = CompatibleSanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        )
        #pipeline.vae=AutoencoderDC.from_pretrained("")
    try:
        pipeline.safety_checker=None
    except Exception as err:
        accelerator.print("tried to set safety checker to None",err)

    if type(pipeline.scheduler)==SCMScheduler:
        pipeline.scheduler=CompatibleSCMScheduler.from_config(pipeline.scheduler.config)
    elif type(pipeline.scheduler)==DEISMultistepScheduler:
        pipeline.scheduler=CompatibleDEISMultistepScheduler.from_config(pipeline.scheduler.config)
    elif type(pipeline.scheduler)==FlowMatchEulerDiscreteScheduler:
        pipeline.scheduler=CompatibleFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)


    accelerator.print(pipeline.scheduler)

    for attribute in ["add_noise","get_velocity","step"]:
        if getattr(pipeline.scheduler,attribute,None) is not None:
            print(f"scheduler {attribute} exists")
        else:
            print(f"scheduler {attribute} does not exist ")


    vae=pipeline.vae
    if args.pipeline=="sana":
        denoising_model=pipeline.transformer
    else:
        denoising_model=pipeline.unet
        pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        accelerator.print(pipeline.scheduler)
        pipeline.unet.encoder_hid_proj=None
    text_encoder=pipeline.text_encoder
    scheduler=pipeline.scheduler
    

    

    #pipeline.requires_grad_(False)
    embedding_list=[]
    text_list=[]
    image_list=[]
    prompt_list=[]
    shuffled_row_list=[row for row in raw_data]

    with torch.no_grad():
        for i,row in enumerate(shuffled_row_list):
            if i==args.limit:
                break
            before_objects=find_cuda_objects()
            image=row["image"]
            
            
            
            if "embedding" in row:
                #print(row["embedding"])
                np_embedding=np.array(row["embedding"])[-1]
                #print("np_embedding",np_embedding.shape)
                embedding=torch.from_numpy(np_embedding)
                #print("embedding",embedding.size())
                #real_embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image)).unsqueeze(0)
                #print("real embedding",real_embedding.size())
            else:
                #this should NOT be normalized or transformed
                embedding=embedding_util.embed_img_tensor(embedding_util.transform_image(image))

            if args.pca_project:
                projection=pca_object.transform(embedding.cpu())
                embedding=torch.Tensor(pca_object.inverse_transform(projection))

            image=pipeline.image_processor.preprocess(image)[0]
            
            image_list.append(image.squeeze(0))
            #print(embedding.size())
            embedding=embedding.to("cpu") #.squeeze()
            embedding_list.append(embedding)
            accelerator.free_memory()
            torch.cuda.empty_cache()

            try:
                text=row["text"]
            except:
                text=row["label"]
            if type(text)==str:
                prompt=text
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
            else:
                np_text=np.array(text)
                text=torch.from_numpy(np_text)
                prompt=row["prompt"]
            text=text.to("cpu").squeeze(0)
            if i ==1:
                accelerator.print("text size",text.size(),"embedding size",embedding.size(),"img size",image.size(),"latent size")
            text_list.append(text)
            prompt_list.append(prompt)
            #print(get_gpu_memory_usage())
            #print("gpu objects:",len(find_cuda_objects()))
            after_objects=find_cuda_objects()
            delete_unique_objects(after_objects,before_objects)
            #print("grads",len(find_cuda_tensors_with_grads()))
    
    accelerator.print("prompt list",len(prompt_list))
    accelerator.print("image_list",len(image_list))
    accelerator.print("text_list",len(text_list))
    accelerator.print("embedding list",len(embedding_list))


    if args.hyperplane:
        classification_data=load_dataset(args.classification_data,split="train")
        classification_data=classification_data.filter(lambda row: row["positives"]>=args.positive_threshold)
        classification_data=list(classification_data)
        accelerator.print([row["label"] for row in classification_data])
        label_real_image_dict={row["label"]:[] for row in classification_data}
        label_fake_image_dict={row["label"]:[] for row in classification_data}

        tagged_data=load_dataset(args.tagged_data,split="train")
        tagged_data=tagged_data.cast_column("image",datasets.Image())
        for row in tagged_data:
            if row["tag"] in label_real_image_dict:
                label_real_image_dict[row["tag"]].append( pipeline.image_processor.preprocess( row["image"]))
        X=[row["embedding"][0] for row in tagged_data]

        scaler =StandardScaler()
        scaler.fit(X)
        model_dict={
            "SVC":LinearSVC,
            "SGD":SGDClassifier
        }
        classifier_model_constructor=model_dict[args.classifier_type]

        label_list=[]
        label_plane_list=[]

        for i, _e in enumerate(embedding_list):
            label_list.append(classification_data[i%len(classification_data)]["label"])
            label_plane_list.append(torch.from_numpy( classification_data[i%len(classification_data)][f"weight_{args.classifier_type}"]/scaler.scale_).unsqueeze(0))

    do_classifier_free_guidance=args.do_classifier_free_guidance
    if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
        do_classifier_free_guidance=True


    for component in [vae,text_encoder]:
        component.requires_grad_(False)
        component.to("cpu")

    if args.pipeline=="sana":
        unconditioned_text,unconditioned_text_attention_mask=pipeline.encode_prompt(
                                       prompt= " ",
                                        device="cpu", #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= do_classifier_free_guidance,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        negative_text_embeds=None
    else:
        unconditioned_text,negative_text_embeds=pipeline.encode_prompt(
                                       prompt= " ",
                                        device="cpu", #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= do_classifier_free_guidance,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )
        unconditioned_text_attention_mask=None
    
    
    
    for i in range(len(text_list)):
        if random.random()<=args.uncaptioned_frac:
            text_list[i]=unconditioned_text.squeeze(0).clone().detach()

    
    fake_image=torch.rand((1,3,args.image_size,args.image_size))
    fake_embedding=embedding_util.embed_img_tensor(fake_image)
    embedding_dim=fake_embedding.size()[-1]

    
    
    denoising_model.requires_grad_(False)
    if args.disable_projection_adapter:
        use_projection=False
    else:
        use_projection=True

    cross_attention_dim=args.cross_attention_dim
    if args.identity_adapter:
        cross_attention_dim=embedding_dim//args.num_image_text_embeds
    intermediate_embedding_dim=args.intermediate_embedding_dim
    if args.disable_projection_adapter:
        intermediate_embedding_dim=embedding_dim

    if use_projection and args.identity_adapter:
        accelerator.print("use_projection and args.identity_adapter are both true")
        

    if args.pipeline=="sana":
        prepare_ip_adapter(pipeline.transformer,accelerator.device,torch_dtype,cross_attention_dim)
    
    replace_ip_attn(denoising_model,
                    embedding_dim,
                    intermediate_embedding_dim,
                    cross_attention_dim,
                    args.num_image_text_embeds,
                    use_projection,args.identity_adapter,args.deep_to_ip_layers)
    #print("image projection",unet.encoder_hid_proj.multi_ip_adapter.image_projection_layers[0])
    start_epoch=1
    if args.load:
        try:
            denoising_model.load_state_dict(torch.load(save_path,weights_only=True),strict=False)
            with open(config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]

            accelerator.print("loaded from ",save_path)
        except Exception as e:
            accelerator.print("couldnt load locally")
            accelerator.print(e)
    if args.load_hf:    
        try:
            pretrained_weights_path=api.hf_hub_download(args.name,WEIGHTS_NAME,force_download=True)
            pretrained_config_path=api.hf_hub_download(args.name,CONFIG_NAME,force_download=True)
            denoising_model.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
            with open(pretrained_config_path,"r") as f:
                data=json.load(f)

            accelerator.print("loaded from  ",pretrained_weights_path)
        except Exception as e:
            accelerator.print("couldnt load from hf")
            accelerator.print(e)
    attn_layer_list=[p for (name,p ) in get_modules_of_types(denoising_model,IPAdapterAttnProcessor2_0)]
    attn_layer_list.append( denoising_model.encoder_hid_proj)
    accelerator.print("len attn_layers",len(attn_layer_list))
    for layer in attn_layer_list:
        layer.requires_grad_(True)


    accelerator.print("before ",denoising_model.config.sample_size)
    denoising_model.config.sample_size=args.image_size // pipeline.vae_scale_factor
    accelerator.print("after", denoising_model.config.sample_size)
    
    ratios=(args.train_split,(1.0-args.train_split)/2.0,(1.0-args.train_split)/2.0)
    accelerator.print('train/test/val',ratios)
    #batched_embedding_list= embedding_list #make_batches_same_size(embedding_list,args.batch_size)
    embedding_list,test_embedding_list,val_embedding_list=split_list_by_ratio(embedding_list,ratios)
    
    image_list,test_image_list,val_image_list=split_list_by_ratio(image_list,ratios)
    text_list,test_text_list,val_text_list=split_list_by_ratio(text_list,ratios)


    prompt_list,test_prompt_list,val_prompt_list=split_list_by_ratio(prompt_list,ratios)

    if args.hyperplane:
        _,test_label_list,_=split_list_by_ratio(label_list,ratios)
        _,test_label_plane_list,_=split_list_by_ratio(label_plane_list,ratios)

    if args.real_test_prompts:
        real_test_prompt_list=[
           ' in the jungle',
            ' in the snow',
            ' on the beach',
            ' on a cobblestone street',
            ' on top of pink fabric',
            ' on top of a wooden floor',
            ' with a city in the background',
            ' with a mountain in the background',
            ' with a blue house in the background',
            ' on top of a purple rug in a forest',
            ' with a wheat field in the background',
            ' with a tree and autumn leaves in the background',
            ' with the Eiffel Tower in the background',
            ' floating on top of water',
            ' floating in an ocean of milk',
            ' on top of green grass with sunflowers around it',
            ' on top of a mirror',
            ' on top of the sidewalk in a crowded street',
            ' on top of a dirt road',
            ' on top of a white rug',
        ]
        test_real_text_embedding_list=[
            pipeline.encode_prompt(
                                       prompt= test_prompt,
                                        device="cpu", #accelerator.device,
                                       num_images_per_prompt= 1,
                                       do_classifier_free_guidance= do_classifier_free_guidance,
                                        negative_prompt="blurry, low quality",
                                        prompt_embeds=None,
                                        negative_prompt_embeds=None,
                                        #lora_scale=lora_scale,
                                )[0] for test_prompt in real_test_prompt_list
        ]

        indices=[random.randint(0,len(real_test_prompt_list)-1) for _ in test_prompt_list]

        test_prompt_list=[real_test_prompt_list[i] for i in indices]
        test_text_list=[test_real_text_embedding_list[i] for i in indices]


    accelerator.print("prompt list",len(prompt_list))
    accelerator.print("image_list",len(image_list))
    accelerator.print("text_list",len(text_list))
    accelerator.print("embedding list",len(embedding_list))

    if args.generic_test_prompts:
        generic_dataset=load_dataset("jlbaker361/test_prompts",split="train")
        generic_tensor_list=[torch.from_numpy(np.array(row["text_embedding"])[0]) for row in generic_dataset]
        generic_str_list=[row["prompt"] for row in generic_dataset]

        for k in range(len(test_prompt_list)):
            #test_prompt_list[k]=generic_str_list[k%len(generic_str_list)]
            test_text_list[k]=generic_tensor_list[k%len(generic_str_list)]

    if args.zeros or args.increasing_scale:
        pipeline.set_ip_adapter_scale(0.0)

    for name, data_list in zip(["train","test","val"],[image_list,test_image_list,val_image_list]):
        accelerator.print(f"{name} has {len(data_list)} samples ")
        if len(data_list):
            accelerator.print("ZERO LEN data partition- this will cause errors")
    
    train_dataset=CustomDataset(image_list,embedding_list,text_list,prompt_list=prompt_list)
    val_dataset=CustomDataset(val_image_list,val_embedding_list,val_text_list,prompt_list=val_prompt_list)
    test_dataset=CustomDataset(test_image_list,test_embedding_list,test_text_list,prompt_list=test_prompt_list)
    if args.hyperplane:
        test_dataset=CustomDataset(test_image_list,test_embedding_list,
                                   test_text_list,prompt_list=test_prompt_list,
                                   label_list=test_label_list, label_plane_list=test_label_plane_list)

    for dataset_batch in train_dataset:
        break

    accelerator.print("dataset batch",type(dataset_batch))

    train_loader=DataLoader(train_dataset,batch_size=args.batch_size,drop_last=True)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size)
    test_loader=DataLoader(test_dataset,args.batch_size)


    for train_batch in train_loader:
        break
    for val_batch in val_loader:
        break
    for test_batch in test_loader:
        break

    accelerator.print("train batch",type(train_batch))
    accelerator.print("val batch",type(val_batch))
    accelerator.print("test batch",type(test_batch))

    params=list(set([p for p in denoising_model.parameters() if p.requires_grad]+[p for p in denoising_model.encoder_hid_proj.parameters() if p.requires_grad]))

    accelerator.print("trainable params: ",len(params))
    for i in range(accelerator.num_processes):
        if accelerator.process_index == i:
            print(f"Rank {i} checkpoint")
            torch.cuda.synchronize()
        accelerator.wait_for_everyone()


    if args.vanilla:
        denoising_model=denoising_model.to(device)

    #if args.training_type=="reward":
    vae=vae.to(denoising_model.device)
    
    #time_embedding=denoising_model.time_embedding.to(denoising_model.device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    if args.fsdp:
        clip_model.logit_scale = torch.nn.Parameter(torch.tensor([clip_model.config.logit_scale_init_value]))
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    fid = FrechetInceptionDistance(feature=2048,normalize=True)
    accelerator.wait_for_everyone()
    clip_model,clip_processor,fid,denoising_model,vae,scheduler=accelerator.prepare(clip_model,clip_processor,fid,denoising_model,vae,scheduler)
    if hasattr(denoising_model,"post_quant_conv"):
        post_quant_conv=denoising_model.post_quant_conv.to(denoising_model.device)
        post_quant_conv=accelerator.prepare(post_quant_conv)
        vae.post_quant_conv=post_quant_conv
    if hasattr(denoising_model,"time_embedding"):
        time_embedding=denoising_model.time_embedding.to(denoising_model.device)
        time_embedding=accelerator.prepare(time_embedding)
        denoising_model.time_embedding=time_embedding

    if hasattr(denoising_model, "patch_embed"):
        patch_embed=denoising_model.patch_embed.to(denoising_model.device, torch_dtype)
        patch_embed=accelerator.prepare(patch_embed)
        denoising_model.patch_embed=patch_embed

    if hasattr(denoising_model,"caption_projection"):
        caption_projection=denoising_model.caption_projection.to(denoising_model.device, torch_dtype)
        caption_projection=accelerator.prepare(caption_projection)
        denoising_model.caption_projection=caption_projection

    if hasattr(denoising_model,"encoder_hid_proj"):
        encoder_hid_proj=denoising_model.encoder_hid_proj.to(denoising_model.device)
        encoder_hid_proj=accelerator.prepare(encoder_hid_proj)
        denoising_model.encoder_hid_proj=encoder_hid_proj
    accelerator.wait_for_everyone()
    train_loader,test_loader,val_loader=accelerator.prepare(train_loader,test_loader,val_loader)
    accelerator.wait_for_everyone()
    #train_loader=accelerator.prepare_data_loader(train_loader,True)
    try:
        register_fsdp_forward_method(vae,"decode")
        accelerator.print("registered")
    except Exception as e:
        accelerator.print('register_fsdp_forward_method',e)
    
    if args.pipeline=="sana":
        pipeline.transformer=denoising_model
    else:
        pipeline.unet=denoising_model
    pipeline.vae=vae

    for loader in [test_loader,val_loader,train_loader]:
        print(type(loader))

    

    def logging(data_loader,pipeline,baseline:bool=False,auto_log:bool=True,clip_model:CLIPModel=clip_model):
        before_objects=find_cuda_objects()
        metrics={}
        difference_list=[]
        embedding_difference_list=[]
        clip_alignment_list=[]
        image_list=[]
        fake_image_list=[]

        if args.pipeline=="lcm_post_lora":
            pipeline.enable_lora()
        pipeline=pipeline.to(accelerator.device)
        pipeline.vae=pipeline.vae.to(accelerator.device)
        pipeline.text_encoder=pipeline.text_encoder.to(accelerator.device)
        if hasattr(pipeline,"unet"):
            pipeline.unet.time_embedding=pipeline.unet.time_embedding.to(accelerator.device)
            pipeline.unet.time_embedding.linear_1.weight=pipeline.unet.time_embedding.linear_1.weight.to(accelerator.device)
            pipeline.unet.time_embedding.linear_1.bias=pipeline.unet.time_embedding.linear_1.bias.to(accelerator.device)

            pipeline.unet.time_embedding.linear_2.weight=pipeline.unet.time_embedding.linear_2.weight.to(accelerator.device)
            pipeline.unet.time_embedding.linear_2.bias=pipeline.unet.time_embedding.linear_2.bias.to(accelerator.device)

            if getattr(pipeline.unet.time_embedding, "cond_proj",None) is not None:
                pipeline.unet.time_embedding.cond_proj.weight=pipeline.unet.time_embedding.cond_proj.weight.to(accelerator.device)
        
        for b,batch in enumerate(data_loader):
            if batch==None:
                continue
            
            for k,v in batch.items():
                if type(v)==torch.Tensor:
                    batch[k]=v.to(accelerator.device,torch_dtype)
            image_batch=batch["image"]
            text_batch=batch["text"]
            embeds_batch=batch["embeds"]
            if args.zeros:
                embeds_batch=torch.zeros(embeds_batch.size())
            prompt_batch=batch["prompt"]
            
            if len(image_batch.size())==3:
                image_batch=image_batch.unsqueeze(0)
                text_batch=text_batch.unsqueeze(0)
                embeds_batch=embeds_batch.unsqueeze(0)
            batch_size=image_batch.size()[0]
            image_embeds=embeds_batch #.unsqueeze(0)
            do_denormalize= [True] * batch_size
            '''if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
                batched_negative_prompt_embeds=negative_text_embeds.expand((batch_size, -1,-1)).to(text_batch.device)
                negative_image_embeds=torch.zeros(image_embeds.size(),device=image_embeds.device)
                image_embeds=[torch.cat([negative_image_embeds,image_embeds],dim=0)]
            else:
                batched_negative_prompt_embeds=None
                image_embeds=[image_embeds]'''
            
            if b==0:
                if hasattr(pipeline,"unet"):
                    print("unet",pipeline.unet.device,"time embedding linear 1",pipeline.unet.time_embedding.linear_1.weight.device, )
                
                accelerator.print("testing","images",image_batch.size(),"text",text_batch.size(),"embeds",embeds_batch.size())
                accelerator.print("testing","images",image_batch.device,"text",text_batch.device,"embeds",embeds_batch.device)
                accelerator.print("testing","images",image_batch.dtype,"text",text_batch.dtype,"embeds",embeds_batch.dtype)
                accelerator.print("prompt",prompt_batch)
                '''if args.pipeline=="lcm_post_lora" or args.pipeline=="lcm_pre_lora":
                    accelerator.print("testing","negstive",batched_negative_prompt_embeds.size(),batched_negative_prompt_embeds.device)'''
            #image_batch=torch.clamp(image_batch, 0, 1)
            real_pil_image_set=pipeline.image_processor.postprocess(image_batch,"pil",do_denormalize)
            
            if baseline:
                baseline_prompt_batch=[p + args.baseline_prompt for p in prompt_batch]
                #ip_adapter_image=F_v2.resize(image_batch, (224,224))
                fake_image=torch.stack([pipeline( baseline_prompt_batch,
                                                    num_inference_steps=args.num_inference_steps,
                                                 #prompt_embeds=text_batch,
                                                 ip_adapter_image=ip_adapter_image, 
                                                 #negative_prompt_embeds=batched_negative_prompt_embeds,
                                                 output_type="pt",height=args.image_size,width=args.image_size).images[0] for ip_adapter_image in real_pil_image_set])
            else:
                if args.cfg_embedding:
                    generator=torch.Generator(accelerator.device)
                    generator.manual_seed(123)
                    null_prompt_image=pipeline([" " for _ in range(batch_size)],
                                    num_inference_steps=args.num_inference_steps,
                                    do_classifier_free_guidance=False,
                                    generator=generator,
                                    #prompt_embeds=text_batch,
                                    ip_adapter_image_embeds=image_embeds,
                                    #negative_prompt_embeds=batched_negative_prompt_embeds,
                                    output_type="pt",height=args.image_size,width=args.image_size,
                                    #decreasing_scale=args.decreasing_scale,increasing_scale=args.increasing_scale,start=args.ip_start,end=args.ip_end
                                    ).images
                    generator=torch.Generator(accelerator.device)
                    generator.manual_seed(123)
                    prompt_image=pipeline(prompt_batch,
                                    num_inference_steps=args.num_inference_steps,
                                    do_classifier_free_guidance=False,
                                    generator=generator,
                                    #prompt_embeds=text_batch,
                                    ip_adapter_image_embeds=image_embeds,
                                    #negative_prompt_embeds=batched_negative_prompt_embeds,
                                    output_type="pt",height=args.image_size,width=args.image_size,
                                    #decreasing_scale=args.decreasing_scale,increasing_scale=args.increasing_scale,start=args.ip_start,end=args.ip_end
                                    ).images
                    
                    null_embedding=embedding_util.embed_img_tensor(null_prompt_image)
                    prompted_embedding=embedding_util.embed_img_tensor(prompt_image)

                    cfg_embedding=prompted_embedding-null_embedding
                    image_embeds=args.cfg_weight*cfg_embedding 
                    image_embeds=image_embeds+embeds_batch
                    
                    #print("embeds size",image_embeds.size())
                    
                if args.hyperplane:
                    label_batch=batch["label"]
                    label_plane=batch["label_plane"].to(image_embeds.device)
                    image_embeds=image_embeds+args.hyperplane_coefficient * label_plane

                image_embeds=[image_embeds]
                
                fake_image=pipeline(prompt_batch,
                                    num_inference_steps=args.num_inference_steps,
                                    do_classifier_free_guidance=do_classifier_free_guidance,
                                    #prompt_embeds=text_batch,
                                    ip_adapter_image_embeds=image_embeds,
                                    #negative_prompt_embeds=batched_negative_prompt_embeds,
                                    output_type="pt",height=args.image_size,width=args.image_size,decreasing_scale=args.decreasing_scale,increasing_scale=args.increasing_scale
                                    ,start=args.ip_start,end=args.ip_end).images
            
            #normal_image_set=pipeline(prompt_embeds=text_batch,output_type="pil").images
            image_batch=F_v2.resize(image_batch, (args.image_size,args.image_size))
            #print("img vs real img",fake_image.size(),image_batch.size())
            #image_embeds.to("cpu")
            image_batch=image_batch.to(fake_image.device)

            difference_list.append(F.mse_loss(fake_image,image_batch).cpu().detach().item())


            embedding_real=embedding_util.embed_img_tensor(image_batch)
            embedding_fake=embedding_util.embed_img_tensor(fake_image)
            embedding_difference_list.append(F.mse_loss(embedding_real,embedding_fake).cpu().detach().item())

            image_list.append(image_batch.cpu())
            fake_image_list.append(fake_image.cpu())
            
            
            
            pil_image_set=pipeline.image_processor.postprocess(fake_image,"pil",do_denormalize)
            pil_image_set_unnorm=pipeline.image_processor.postprocess(fake_image,"pil",[False]*batch_size)
            


            
            inputs = clip_processor(
                text=prompt_batch, images=pil_image_set_unnorm, return_tensors="pt", padding=True
            )
            for k,v in inputs.items():
                inputs[k]=v.to(clip_model.device)

            outputs = clip_model(**inputs)
            clip_text_embeds=outputs.text_embeds
            clip_image_embeds=outputs.image_embeds
            clip_difference=F.mse_loss(clip_image_embeds,clip_text_embeds)
            
            clip_alignment_list.append(clip_difference.cpu().detach().item())

            if args.hyperplane:
                for l,img in zip(label_batch,fake_image):
                    label_fake_image_dict[l].append(pipeline.image_processor.preprocess(img))

            for k,(pil_image,real_pil_image,pil_image_unnorm,prompt) in enumerate(zip(pil_image_set,real_pil_image_set,pil_image_set_unnorm,prompt_batch)):
                concat_image=concat_images_horizontally([real_pil_image,pil_image_unnorm,pil_image])
                if args.hyperplane:
                    prompt+=f" {label_batch[k]} "
                metrics[prompt.replace(",","").replace(" ","_").strip()]=wandb.Image(concat_image)
        #pipeline.scheduler =  DEISMultistepScheduler.from_config(pipeline.scheduler.config)
        
        metrics["difference"]=np.mean(difference_list)
        metrics["embedding_difference"]=np.mean(embedding_difference_list)
        metrics["text_alignment"]=np.mean(clip_alignment_list)
            #print("size",torch.cat(image_list).size())
        if args.fid:
            start=time.time()
            fid_dtype=next(fid.inception.parameters()).dtype
            fid_device=next(fid.inception.parameters()).device
            fid.update(torch.cat(image_list).to(fid_device,fid_dtype),real=True)
            fid.update(torch.cat(fake_image_list).to(fid_device,fid_dtype),real=False)
            metrics["fid"]=fid.compute().cpu().detach().item()
            end=time.time()
            print("fid elapsed ",end-start)
        else:
            metrics["fid"]=0.0

        if args.hyperplane:
            label_fid_list=[]
            for k in label_fake_image_dict.keys():
                if k in label_real_image_dict:
                    fid_dtype=next(fid.inception.parameters()).dtype
                    fid_device=next(fid.inception.parameters()).device
                    label_real_image_tensor_list=label_real_image_dict[k]
                    label_fake_image_tensor_list= label_fake_image_dict[k]
                    if len(label_fake_image_tensor_list) >1 and len(label_real_image_tensor_list)>1:
                        fid.update(torch.cat(label_real_image_tensor_list).to(fid_device,fid_dtype), real=True)
                        fid.update(torch.cat(label_fake_image_tensor_list).to(fid_device,fid_dtype), real = False)
                        difference=fid.compute().cpu().detach().item()
                        label_fid_list.append(difference)
                        accelerator.print(k,difference )
            metrics["label_fid"]=np.mean(label_fid_list)



        if auto_log:
            accelerator.log(metrics)
        if args.pipeline=="lcm_post_lora":
            pipeline.disable_lora()
        
        after_objects=find_cuda_objects()
        delete_unique_objects(before_objects,after_objects)
        return metrics

    training_start=time.time()
    
    '''val_start=time.time()
    before_objects=find_cuda_objects()
    with torch.no_grad():

        start=time.time()
        clip_model=clip_model.to(denoising_model.device)
        val_metrics=logging(val_loader,pipeline,clip_model=clip_model)
        new_metrics={}
        for k,v in val_metrics.items():
            new_metrics["val_"+k]=v
            accelerator.print("\tTEST",k,v)
        accelerator.log(new_metrics)
        clip_model=clip_model.cpu()
        end=time.time()
        persistent_fid_list.append(val_metrics["fid"])
        persistent_text_alignment_list.append(val_metrics["text_alignment"])
    after_objects=find_cuda_objects()
    delete_unique_objects(after_objects,before_objects)'''
    

    training_end=time.time()
    accelerator.print(f"total trainign time = {training_end-training_start}")
    accelerator.free_memory()
    clip_model=clip_model.to(denoising_model.device)
    metrics=logging(test_loader,pipeline,auto_log=False)
    new_metrics={}
    for k,v in metrics.items():
        new_metrics["test_"+k]=v
        accelerator.print("\tTEST",k,v)
    accelerator.log(new_metrics)

    if args.pipeline!="sana":
        if args.pipeline=="lcm":
            baseline_pipeline=DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7",device=accelerator.device,torch_dtype=torch_dtype)
        else:
            baseline_pipeline=DiffusionPipeline.from_pretrained("Lykon/dreamshaper-7",device=accelerator.device,torch_dtype=torch_dtype)
            
            baseline_pipeline.load_lora_weights(adapter_id)
            baseline_pipeline.fuse_lora()
        baseline_pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        try:
            baseline_pipeline.safety_checker=None
        except Exception as err:
            accelerator.print("tried to set safety checker to None",err)
        b_unet=baseline_pipeline.unet.to(device,torch_dtype)
        b_text_encoder=baseline_pipeline.text_encoder.to(device,torch_dtype)
        b_vae=baseline_pipeline.vae.to(device,torch_dtype)
        b_image_encoder=baseline_pipeline.image_encoder.to(device,torch_dtype)

        b_unet,b_text_encoder,b_vae,b_image_encoder=accelerator.prepare(b_unet,b_text_encoder,b_vae,b_image_encoder)
        baseline_pipeline.unet=b_unet
        baseline_pipeline.text_encoder=b_text_encoder
        baseline_pipeline.vae=b_vae
        baseline_pipeline.image_encoder=b_image_encoder
        baseline_metrics=logging(test_loader,baseline_pipeline,baseline=True)
        new_metrics={}
        for k,v in baseline_metrics.items():
            new_metrics["baseline_"+k]=v
            accelerator.print("\tBASELINE",k,v)
        accelerator.log(new_metrics)
    accelerator.wait_for_everyone()
    accelerator.log({"finished":True})
        

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