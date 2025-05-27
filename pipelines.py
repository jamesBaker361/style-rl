from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
from diffusers import DiffusionPipeline,LatentConsistencyModelPipeline
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import retrieve_timesteps
import torch
from typing import Union,Any,Optional,Callable,List,Dict
import torch.utils.checkpoint as checkpoint
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.outputs import BaseOutput
from dataclasses import dataclass
import PIL
import numpy as np
from copy import deepcopy
import random

def register_evil_twin(pipeline:DiffusionPipeline,scale:float):
    unet=pipeline.unet
    pipeline.evil_twin_unet=deepcopy(unet)
    pipeline.evil_twin_unet.to(unet.device,unet.dtype)
    pipeline.evil_twin_guidance_scale=scale
    return pipeline.evil_twin_unet

@dataclass
class CustomStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
    latents: Optional[torch.Tensor]
    

class CompatibleLatentConsistencyModelPipeline(LatentConsistencyModelPipeline):
    def check_inputs(self, prompt,
                      height, 
                      width, 
                      callback_steps, 
                      negative_prompt=None,
                      prompt_embeds = None, 
                      negative_prompt_embeds=None,
                      ip_adapter_image=None, 
                      ip_adapter_image_embeds=None, 
                      callback_on_step_end_tensor_inputs=None):
        return super().check_inputs(prompt, height, width, callback_steps, prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, callback_on_step_end_tensor_inputs)
    
    @torch.no_grad()
    def __call__(
        self,
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
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

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
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

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

        # 5. Prepare latent variable
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.unet.dtype,
            device,
            generator,
            latents,
        )
        #print("compatiable latenst size",latents.size())
        bs = batch_size * num_images_per_prompt

        # 6. Get Guidance Scale Embedding
        # NOTE: We use the Imagen CFG formulation that StableDiffusionPipeline uses rather than the original LCM paper
        # CFG formulation, so we need to subtract 1 from the input guidance_scale.
        # LCM CFG formulation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond), (cfg_scale > 0.0 using CFG)
        w = torch.tensor(self.guidance_scale - 1).repeat(bs)
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
            device=device, dtype=latents.dtype
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )
                added_cond_kwargs={"image_embeds":image_embeds}
        else:
            added_cond_kwargs={}

        # 8. LCM MultiStep Sampling Loop:
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        latents_clone=latents.clone()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                #print(f"step {i}/num_inference_steps")
                latents = latents.to(prompt_embeds.dtype)

                # model prediction (v-prediction, eps, x)
                model_pred = self.unet(
                    latents,
                    t,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                if hasattr(self,"evil_twin_unet"):
                    #print("evil twin >:)")
                    evil_twin_model_pred=self.evil_twin_unet(
                        latents,
                        t,
                        timestep_cond=w_embedding,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    model_pred=evil_twin_model_pred + self.evil_twin_guidance_scale* (model_pred-evil_twin_model_pred)
                # compute the previous noisy sample x_t -> x_t-1
                #print('model_pred.device',model_pred.device,'t device',t.device,'latents',latents.device)
                t=t.to(model_pred.device)
                latents=latents.to(model_pred.device)
                latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)
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

                

        denoised = denoised.to(prompt_embeds.dtype)
        has_nsfw_concept = None
        if not output_type == "latent":
            print("denoised ",denoised.size())
            image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
            
        else:
            image = denoised

        do_denormalize = [True] * image.shape[0]
        try:
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        except RuntimeError:
            image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return CustomStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept,latents=latents_clone)
    

    def call_with_grad(
        self,
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
        **kwargs,
    ):
        
        #print("rgb with grad before",len(find_cuda_objects()))
        ##print("407 added condkwagrs",added_cond_kwargs)
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
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
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )
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
            w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=self.unet.config.time_cond_proj_dim).to(
                device=device, dtype=self.unet.dtype
            )

            # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)
            #print("rgb with grad no grad",len(find_cuda_objects()))

        
            # 5. Prepare latent variable
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
                latents=latents.to(self.unet.device)
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
                #print(f"step {i}/num_inference_steps")
                #latents = latents.to(prompt_embeds.dtype)
                #print("latents size",latents.size())
                # model prediction (v-prediction, eps, x)
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
                print("model pred grad",model_pred.requires_grad)
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
                print('model_pred.device',model_pred.device,'t device',t.device,'latents',latents.device)
                t=t.to(model_pred.device)
                #latents=latents.to(model_pred.device)
                latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)
                print("latents grad",latents.requires_grad)
                print("model pred grad",model_pred.requires_grad)
                #print("loop latents denoised",len(find_cuda_objects()))   
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
        print("denoised grad",denoised.requires_grad)
        has_nsfw_concept = None
        if not output_type == "latent":
            image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
            
        else:
            image = denoised

        do_denormalize = [True] * image.shape[0]
        try:
            image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        except RuntimeError:
            image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        #print("called with grad",len(find_cuda_objects()))
        return CustomStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept,latents=latents_copy)
    

    def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds = None, negative_prompt_embeds = None, lora_scale = None, clip_skip = None):
        prompt_embeds_tuple= self.encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)
        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds
    
    def run_safety_checker(self,image,*args,**kwargs):
        return image,None
    
#class PromptCompatibleLatentConsistencyModelPipeline(CompatibleLatentConsistencyModelPipeline):
    def register_prompt_model(self,prompt_model:torch.nn.Module,src_entity:torch.Tensor,num_image_text_embeds:int=16):
        self.prompt_model=prompt_model
        self.src_entity=src_entity

    def register_encoder_hid_proj(self,encoder_hid_proj:torch.nn.Module,src_embeds:torch.Tensor):
        self.src_embeds=src_embeds
        self.unet.config.encoder_hid_dim_type = "image_proj"
        self.unet.encoder_hid_proj=encoder_hid_proj

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt
    def encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt=None, prompt_embeds = None, negative_prompt_embeds = None, lora_scale = None, clip_skip = None):
        positive,negative= super().encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)
        #print("positive shape", positive.size())
        if hasattr(self, "prompt_model"):
            positive=self.prompt_model(self.src_entity,positive)
        return positive,negative
    
    def get_trainable_layers(self)->tuple:
        unet_parameters=[p for p in self.unet.named_parameters() if p[1].requires_grad]
        other_parameters=[]
        if hasattr(self,"prompt_model"):
            other_parameters=[p for _,p in self.prompt_model.named_parameters()]
        elif hasattr(self,"src_embeds"):
            other_parameters=[p for _,p in self.unet.encoder_hid_proj.named_parameters()]
        return unet_parameters,other_parameters


class KeywordDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self,sd_pipeline:CompatibleLatentConsistencyModelPipeline,keywords:set,use_lora:bool=False,output_type:str="pt",gradient_checkpoint: bool = True,textual_inversion:bool=False):
        self.sd_pipeline=sd_pipeline
        self.keywords=keywords
        self.use_lora=use_lora
        self.output_type=output_type
        self.gradient_checkpoint=gradient_checkpoint
        self.textual_inversion=textual_inversion
        for layer in self.get_trainable_layers():
            layer.requires_grad_(True)

    def get_trainable_layers(self):
        ret=[]
        if self.textual_inversion:
            text_params=[p for _,p in self.sd_pipeline.text_encoder.named_parameters() if p.requires_grad]
            print("len text parpams",len(text_params))
            ret+=text_params
        unet_parameters,other_parameters=self.sd_pipeline.get_trainable_layers()
        print('len(unet_parameters)',len(unet_parameters))
        if len(self.keywords)==0:
            ret+= [p for _,p in unet_parameters if p.requires_grad]
        for key in self.keywords:
            for name,p in unet_parameters:
                if name.find(key)!=-1 and p.requires_grad:
                    ret.append(p)
        return ret+other_parameters

    def rgb_with_grad(self,*args,**kwargs):
        #print("rgb with grad",len(find_cuda_objects()))
        kwargs["output_type"]=self.output_type
        kwargs["gradient_checkpoint"]=self.gradient_checkpoint
        if type(self.sd_pipeline)==CompatibleLatentConsistencyModelPipeline:
            return self.sd_pipeline.call_with_grad(*args,**kwargs)
        else:
            return super().rgb_with_grad(*args,**kwargs)