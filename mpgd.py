from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial
from pipelines import *
from diffusers.utils.loading_utils import load_image
from embedding_helpers import EmbeddingUtil
from main_pers import concat_images_horizontally
from gpu_helpers import get_gpu_memory_usage
from custom_scheduler import CompatibleDDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers import DDIMScheduler
import torch
from PIL import Image
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, ToTensor, Compose, Resize



def call_with_grad_and_guidance(
    self:LatentConsistencyModelPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 4,
    original_inference_steps: int = None,
    timesteps: List[int] = None,
    guidance_scale: float = 8.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    truncated_backprop: bool = True,
    truncated_backprop_rand: bool = True,
    gradient_checkpoint: bool = True,
    truncated_backprop_timestep: int = 0,
    truncated_rand_backprop_minmax: tuple = (0, 50),
    fsdp:bool=False, #if fsdp, we have to do some stupid thing where we call the unet once to get model_pred device
    reward_training:bool=False, 
    negative_prompt_embeds:Optional[torch.Tensor]=None, #this does not get used but is only here for compatiabiltiy
    use_resolution_binning:bool=False,
    denormalize_option:bool=True,
    target=None,
    embedding_model: EmbeddingUtil=None,
    **kwargs,
):
    
    #print("rgb with grad before",len(find_cuda_objects()))
    ##print("407 added condkwagrs",added_cond_kwargs)
    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    # 0. Default height and width to unet
    unwrapped_unet = getattr(self.unet, "module", self.unet)
    unwrapped_vae=getattr(self.vae,"module",self.vae)
    height = height or unwrapped_unet.config.sample_size * self.vae_scale_factor
    width = width or unwrapped_unet.config.sample_size * self.vae_scale_factor
    #print("height,width,self.unet.config.sample_size,self.vae_scale_factor",height,width,self.unet.config.sample_size,self.vae_scale_factor)
    with torch.no_grad():
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            None,
            prompt_embeds,
            None,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.unet.device

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            #print("before shape ",ip_adapter_image_embeds[0].size())
            if reward_training:
                image_embeds=ip_adapter_image_embeds
            else:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )
            #print("after shape",image_embeds[0].size())
            added_cond_kwargs={"image_embeds":image_embeds}
        else:
            added_cond_kwargs={}

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        # NOTE: when a LCM is distilled from an LDM via latent consistency distillation (Algorithm 1) with guided
        # distillation, the forward pass of the LCM learns to approximate sampling from the LDM using CFG with the
        # unconditional prompt "" (the empty string). Due to this, LCMs currently do not support negative prompts.
        prompt_embeds, _ = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
        )

        
        bs = batch_size * num_images_per_prompt

        # 6. Get Guidance Scale Embedding
        # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
        # CFG formulation, so we need to subtract 1 from the input guidance_scale.
        # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
        w = torch.tensor(self.guidance_scale - 1).repeat(bs)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=unwrapped_unet.config.time_cond_proj_dim).to(
            device=device, dtype=unwrapped_unet.dtype
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)
        #print("rgb with grad no grad",len(find_cuda_objects()))

    
        # 5. Prepare latent variable
        num_channels_latents = unwrapped_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if fsdp:
            t=timesteps[0]
            model_pred = self.unet(
                    latents,
                    t,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            latents=latents.to(model_pred.device)
        else:
            latents=latents.to(unwrapped_unet.device)
        #print("compatiable latenst size call with grad",latents.size())
    #print("rgb with grad latents",len(find_cuda_objects()))

    #latents.requires_grad_(True)
    
    

    #print("506 added condkwagrs",added_cond_kwargs)
    # 8. LCM MultiStep Sampling Loop:
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    latents_copy=latents.clone()

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if gradient_checkpoint:
                #print("516 added condkwagrs",added_cond_kwargs)
                model_pred = checkpoint.checkpoint(
                    self.unet,
                    latents,
                    t,
                    prompt_embeds,
                    None,
                    w_embedding,
                    None,
                    self.cross_attention_kwargs,
                    added_cond_kwargs,
                    None,None,None,None,
                    False,
                    use_reentrant=False
                )[0]
            else:
                model_pred = self.unet(
                    latents,
                    t,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            #print("loop model_pred",len(find_cuda_objects()))   
            #print("model pred grad post unet",model_pred.requires_grad)
            if truncated_backprop:
                if truncated_backprop_rand:
                    rand_timestep = random.randint(
                        truncated_rand_backprop_minmax[0], truncated_rand_backprop_minmax[1])
                    if i < rand_timestep:
                        model_pred = model_pred.detach()
            else:
                # fixed truncation process
                if i < truncated_backprop_timestep:
                    model_pred = model_pred.detach()
            # compute the previous noisy sample x_t -> x_t-1

            t=t.to(model_pred.device)
            #latents=latents.to(model_pred.device)
            latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)

            guidance_strength=0.0000001
            if target is not None and embedding_model is not None:
                with torch.enable_grad():
                    decoded=self.vae.decode(denoised.clone().detach()).sample
                    decoded.requires_grad_(True)
                    decoded_embedding=embedding_model.embed_img_tensor(decoded)

                    diff=torch.nn.functional.mse_loss(decoded_embedding,target)

                    diff_gradient=torch.autograd.grad(outputs=diff,inputs=decoded)[0]
                    diff_gradient=guidance_strength*diff_gradient
                    
                    usage=get_gpu_memory_usage()
                    torch.cuda.empty_cache()
                    print(usage["allocated_mb"])

                new_denoised=self.vae.encode(decoded+diff_gradient.detach()).latent_dist.sample()

                new_latents=self.scheduler.add_noise(new_denoised,model_pred,t)
                latents, denoised = self.scheduler.step(model_pred, t, new_latents, **extra_step_kwargs, return_dict=False)

  
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                w_embedding = callback_outputs.pop("w_embedding", w_embedding)
                denoised = callback_outputs.pop("denoised", denoised)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    #print("rgb with grad loop",len(find_cuda_objects()))         

    denoised = denoised.to(prompt_embeds.dtype)
    #print("denoised grad",denoised.requires_grad)
    has_nsfw_concept = None
    if not output_type == "latent":

        image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
        if denormalize_option==True:
            do_denormalize = [denormalize_option] * image.shape[0]
            #print("imahe decoded",image.requires_grad)
            try:
                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
            except RuntimeError:
                image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)
        #print("imahe processed",image.requires_grad)
    else:
        image = denoised

    
    

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept,latents_copy)

    #print("called with grad",len(find_cuda_objects()))
    return CustomStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept,latents=latents_copy)

def rescale_grad(
    grad: torch.Tensor, clip_scale, **kwargs
):  # [B, N, 3+5]
    node_mask = kwargs.get('node_mask', None)

    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:  # [B, N, 1]
        scale: torch.Tensor = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)  # [B]
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale  # [B]
        grad = grad * co_ef.view(-1, 1, 1)

    return grad

class StyleCLIP(torch.nn.Module):

    def __init__(self, network, device, target=None):
        super(StyleCLIP, self).__init__()

        self.model = CLIPModel.from_pretrained(network)
        
        processor = AutoProcessor.from_pretrained(network).image_processor

        self.image_size = [processor.crop_size['height'], processor.crop_size['width']]

        self.transforms = Compose([
            Normalize(
                mean=processor.image_mean,
                std=processor.image_std
            ),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(network)

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        if target is not None:
            self.target_embedding = self.get_target_embedding(target)

    @torch.no_grad()
    def get_target_embedding(self, target:Union[str, Image.Image, torch.Tensor]):
        if type(target)==torch.Tensor:
            return self.get_gram_matrix(target)
        if type(target)==str:
            img = Image.open(target).convert('RGB')
        elif type(target)==Image.Image:
            img=target
        image = img.resize(self.image_size, Image.Resampling.BILINEAR)
        #image=ToTensor()(image)
        image = self.transforms(ToTensor()(image)).unsqueeze(0)
        return self.get_gram_matrix(image)

    def get_gram_matrix(self, img:torch.Tensor):
        img = img.to(self.device)
        img = torch.nn.functional.interpolate(img, size=self.image_size, mode='bicubic')
        #img = self.transforms(img)
        # following mpgd
        feats = self.model.vision_model(img, output_hidden_states=True, return_dict=True).hidden_states[2]        
        feats = feats[:, 1:, :]  # [bsz, seq_len, h_dim]
        gram = torch.bmm(feats.transpose(1, 2), feats)
        return gram

    def to_tensor(self, img):
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return self.transforms(ToTensor()(img)).unsqueeze(0)

    def forward(self, x):

        embed = self.get_gram_matrix(x)
        diff = (embed - self.target_embedding).reshape(embed.shape[0], -1)
        similarity = -(diff ** 2).sum(dim=1).sqrt() / 100

        return similarity
    

class TextCLIP(torch.nn.Module):
    def __init__(self, network, device, target=None):
        super(TextCLIP, self).__init__()

        self.model = CLIPModel.from_pretrained(network)
        
        processor = AutoProcessor.from_pretrained(network).image_processor

        self.image_size = [processor.crop_size['height'], processor.crop_size['width']]

        self.transforms = Compose([
            Normalize(
                mean=processor.image_mean,
                std=processor.image_std
            ),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained(network)

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        if target is not None:
            self.target_embedding = self.get_target_embedding(target)

    @torch.no_grad()
    def get_target_embedding(self, target:Union[str,torch.Tensor]):
        if type(target)==torch.Tensor:
            return target
        if type(target)==str:
            inputs = self.tokenizer(target, return_tensors="pt", padding=True, truncation=True)

        for k,v in inputs.items():
            inputs[k]=v.to(self.device)

        text_features= self.model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_gram_matrix(self, img:torch.Tensor):
        img = img.to(self.device)
        img = torch.nn.functional.interpolate(img, size=self.image_size, mode='bicubic')
        #img = self.transforms(img)
        # following mpgd
        feats = self.model.vision_model(img, output_hidden_states=True, return_dict=True).hidden_states[2]        
        feats = feats[:, 1:, :]  # [bsz, seq_len, h_dim]
        gram = torch.bmm(feats.transpose(1, 2), feats)
        return gram

    def to_tensor(self, img):
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        return self.transforms(ToTensor()(img)).unsqueeze(0)

    def forward(self, img):

        img = img.to(self.device)
        img = torch.nn.functional.interpolate(img, size=self.image_size, mode='bicubic')

        image_features = self.model.get_image_features(pixel_values=img)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(image_features,self.target_embedding)

        return similarity
    


def ddim_call_with_guidance(
    self:StableDiffusionPipeline,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end= None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    style_clip:StyleCLIP=None,
    text_clip:TextCLIP=None,
    task:str="style", #could also be text
    stage:str="mid",
    guidance_strength:float=1.0,
    **kwargs,
):
    with torch.no_grad():
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )


        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        original_latents=latents.clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    start=int(0.3*self._num_timesteps)
    end=int(0.7 * self._num_timesteps)
    denoised_list=[]
    log_probs_list=[]
    latents_list=[]
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            with torch.no_grad():
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents,denoised = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)

            if  ((stage=="early" and i < start) or (stage=="mid" and i >= start and i <= end) or (stage=="late" and i >= end)):
                task_model=None
                if style_clip is not None and task=="style":
                    task_model=style_clip
                elif text_clip is not None and task=="text":
                    task_model=text_clip 
                if task_model is not None:
                    with torch.enable_grad():
                        new_denoised=denoised.clone().detach()
                        new_denoised.requires_grad_(True)
                        decoded=self.vae.decode(new_denoised).sample
                        
                        log_probs=task_model(decoded)

                        log_probs_list.append(log_probs.sum())

                        diff_gradient=torch.autograd.grad(outputs=log_probs.sum(),inputs=new_denoised)[0]

                        diff_gradient=rescale_grad(diff_gradient,1.0)
                        
                        diff_gradient=guidance_strength*diff_gradient
                        
                        usage=get_gpu_memory_usage()
                        torch.cuda.empty_cache()



                    #new_denoised=self.vae.encode(decoded+diff_gradient.detach()).latent_dist.sample()

                    latents=denoised-diff_gradient

                    new_noise= original_latents
                    
                    new_latents=self.scheduler.add_noise(latents,new_noise,t)
                    latents, denoised = self.scheduler.step(new_noise, t, new_latents, **extra_step_kwargs, return_dict=False)
            latents_list.append(latents.detach())
            denoised_list.append(denoised)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)


    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        
    else:
        image = latents

    has_nsfw_concept=None

    do_denormalize = [True] * image.shape[0]
    image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image, has_nsfw_concept)
    
    if len(denoised_list)>0:
        denoised_list=self.image_processor.postprocess(torch.cat(denoised_list),output_type=output_type,do_denormalize=[True] * len(denoised_list))

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),denoised_list,log_probs_list,latents_list

    

if __name__=="__main__":
    ddim = DDIMScheduler.from_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
    pipeline=StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",torch_dtype=torch.float16,
                                                     #force_download=True,
                                                     scheduler=ddim).to("cuda")
    #pipeline.do_classifier_free_guidance=False
    #pipeline.scheduler=CompatibleDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.vae.requires_grad_(False)
    dim=512

    def style_grad():
    
        url_dict={
            "starry":"https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
            "anime":"anime.jpg",
        #  "cubism":"cubism.jpg",
        #  "ghibli":"ghibli.jpg",
        # "renn":"rennaissance.jpg"
        }

        for k,v in url_dict.items():
            print("trying to load ",k)
            load_image(v)

        '''target=embedding_model.embed_img_tensor(target_tensor)
        print('target size',target.size())'''

        
        for steps in [10,30,50]:
            generator=torch.Generator(pipeline.unet.device)
            generator.manual_seed(123)
            output,denoised_list,log_probs_list,latents_list=ddim_call_with_guidance(pipeline,"smiling boy",dim,dim,
                                            #target=target,
                                            generator=generator,num_inference_steps=steps,
                                            #embedding_model=embedding_mode
                                            )
            base_image=output.images[0]
            base_image.save(f"images/base_{steps}.png")
            base_denoised_list=concat_images_horizontally(denoised_list)
            base_denoised_list.save(f"images/base_concat_{steps}.png")
            for guidance_strength in [-10,-5,-1,1,5,10]:
                
                for k,v in url_dict.items():
                    for stage in ["early","mid","late"]:
                        target_image=load_image(v)
                        #target_tensor=pipeline.image_processor.preprocess(target_image,dim,dim).to("cuda",dtype=torch.float16,)

                        #embedding_model=EmbeddingUtil(pipeline.unet.device,pipeline.unet.dtype, "clip","key",4)
                        style_clip=StyleCLIP('openai/clip-vit-base-patch16',pipeline.unet.device,target_image)

                        print(k,stage)
                        #print("\t",style_clip.target_embedding)

                        
                        generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        output,denoised_list,log_probs_list,latents_list=ddim_call_with_guidance(pipeline,"smiling boy",dim,dim,
                                                                                                 num_inference_steps=steps,
                                                        style_clip=style_clip,
                                                        #target=target,
                                                        generator=generator,num_inference_steps=steps,
                                                        #embedding_model=embedding_model,
                                                        guidance_strength=guidance_strength,
                                                        stage=stage)
                        
                        print(k,stage,log_probs_list)
                        

                        '''generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        image=ddim_call_with_guidance(pipeline,"cat",dim,dim,generator=generator,num_inference_steps=steps,
                                                    guidance_strength=guidance_strength).images[0]'''

                        '''generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        normal_image=pipeline("cat",dim,dim,generator=generator,num_inference_steps=steps).images[0]'''
                        
                        '''concat_image=concat_images_horizontally([image,grad_image])

                        concat_image.save(f"concat_{guidance_strength}_{steps}.png")'''
                        grad_image=output.images[0]
                        grad_image.save(f"images/mpgd_{guidance_strength}_{steps}_{k}_{stage}.png")
                    '''final_image=output.images[0]
                    final_image.save(f"images/mpgd_{guidance_strength}_{steps}_{k}.png")'''
            print(f"all done {steps} ")

    def text_grad():
        prompt_dict={
            "anime-singleton":"anime",
            "picasso":"picasso",
            "cubism":"cubism",
            "renn":"rennaissance painting"
        }

        for steps in [50]:
            generator=torch.Generator(pipeline.unet.device)
            generator.manual_seed(123)
            output,denoised_list,log_probs_list,latents_list=ddim_call_with_guidance(pipeline,"smiling boy",dim,dim,
                                            #target=target,
                                            generator=generator,num_inference_steps=steps,
                                            #embedding_model=embedding_mode
                                            )
            base_image=output.images[0]
            base_image.save(f"images/base_{steps}.png")
            base_denoised_list=concat_images_horizontally(denoised_list)
            base_denoised_list.save(f"images/base_concat_{steps}.png")
            for guidance_strength in [-10,-5,-1,1,5,10]:
                
                for k,v in prompt_dict.items():
                    for stage in ["early","mid","late"]:
                        #target_tensor=pipeline.image_processor.preprocess(target_image,dim,dim).to("cuda",dtype=torch.float16,)

                        #embedding_model=EmbeddingUtil(pipeline.unet.device,pipeline.unet.dtype, "clip","key",4)
                        text_clip=TextCLIP('openai/clip-vit-base-patch16',pipeline.unet.device,v)

                        print(k,stage)
                        #print("\t",style_clip.target_embedding)

                        
                        generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        output,denoised_list,log_probs_list,latents_list=ddim_call_with_guidance(pipeline,"smiling boy",dim,dim,
                                                        text_clip=text_clip,
                                                        task="text",
                                                        #target=target,
                                                        generator=generator,num_inference_steps=steps,
                                                        #embedding_model=embedding_model,
                                                        guidance_strength=guidance_strength,
                                                        stage=stage)
                        
                        print(k,stage,log_probs_list)
                        

                        '''generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        image=ddim_call_with_guidance(pipeline,"cat",dim,dim,generator=generator,num_inference_steps=steps,
                                                    guidance_strength=guidance_strength).images[0]'''

                        '''generator=torch.Generator(pipeline.unet.device)
                        generator.manual_seed(123)
                        normal_image=pipeline("cat",dim,dim,generator=generator,num_inference_steps=steps).images[0]'''
                        
                        '''concat_image=concat_images_horizontally([image,grad_image])

                        concat_image.save(f"concat_{guidance_strength}_{steps}.png")'''
                        grad_image=output.images[0]
                        grad_image.save(f"images/mpgd_{guidance_strength}_{steps}_{k}_{stage}.png")
                    '''final_image=output.images[0]
                    final_image.save(f"images/mpgd_{guidance_strength}_{steps}_{k}.png")'''
            print(f"all done {steps} ")

    text_grad()
