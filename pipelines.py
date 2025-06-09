from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,DDPOPipelineOutput,DDPOStableDiffusionPipeline
from diffusers import DiffusionPipeline,LatentConsistencyModelPipeline
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import retrieve_timesteps
import torch
from typing import Union,Any,Optional,Callable,List,Dict,Tuple
import torch.utils.checkpoint as checkpoint
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.outputs import BaseOutput
from diffusers.models import ImageProjection
from dataclasses import dataclass
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
import PIL
import numpy as np
from copy import deepcopy
import random
from diffusers import SanaSprintPipeline
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import LCMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN
import warnings
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

def register_evil_twin(pipeline:DiffusionPipeline,scale:float):
    unet=pipeline.unet
    pipeline.evil_twin_unet=deepcopy(unet)
    pipeline.evil_twin_unet.to(unet.device,unet.dtype)
    pipeline.evil_twin_guidance_scale=scale
    return pipeline.evil_twin_unet

def encode_prompt_distributed(self:LatentConsistencyModelPipeline,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        unwrapped_text_encoder=getattr(self.text_encoder,"module",self.text_encoder)
        unwrapped_unet = getattr(self.unet, "module", self.unet)
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(unwrapped_text_encoder, lora_scale)
            else:
                scale_lora_layers(unwrapped_text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )

            if hasattr(unwrapped_text_encoder.config, "use_attention_mask") and unwrapped_text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if unwrapped_text_encoder is not None:
            prompt_embeds_dtype = unwrapped_text_encoder.dtype
        elif unwrapped_unet is not None:
            prompt_embeds_dtype = unwrapped_unet = getattr(self.unet, "module", self.unet).dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(unwrapped_text_encoder.config, "use_attention_mask") and unwrapped_text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if unwrapped_text_encoder is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(unwrapped_text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

def prepare_ip_adapter_image_embeds_distributed(self:LatentConsistencyModelPipeline, 
                                                ip_adapter_image, 
                                                ip_adapter_image_embeds:List[torch.Tensor], 
                                                device, 
                                                num_images_per_prompt:int, 
                                                do_classifier_free_guidance:bool
    ):
        unwrapped_unet = getattr(self.unet, "module", self.unet)
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(unwrapped_unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(unwrapped_unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, unwrapped_unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    negative_image_embeds.append(single_negative_image_embeds)
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

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

        unwrapped_unet = getattr(self.unet, "module", self.unet)
        unwrapped_vae=getattr(self.vae,"module",self.vae)
        # 0. Default height and width to unet
        height = height or unwrapped_unet.config.sample_size * self.vae_scale_factor
        width = width or unwrapped_unet.config.sample_size * self.vae_scale_factor

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

        device = unwrapped_unet.device

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            unet_type=str(type(self.unet))
            if unet_type.find("DistributedDataParallel")!=-1:
                image_embeds=prepare_ip_adapter_image_embeds_distributed(
                    self,
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                    self.do_classifier_free_guidance,
                )
            else:
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
        try:
            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps, original_inference_steps=original_inference_steps
            )
        except TypeError:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device, timesteps
            )

        # 5. Prepare latent variable
        num_channels_latents = unwrapped_unet.config.in_channels
        vae_type=str(type(self.vae))

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            unwrapped_unet.dtype,
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
        w_embedding = self.get_guidance_scale_embedding(w, embedding_dim=unwrapped_unet.config.time_cond_proj_dim).to(
            device=device, dtype=latents.dtype
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, None)

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
            #print("denoised ",denoised.size())
            image = unwrapped_vae.decode(denoised / unwrapped_vae.config.scaling_factor, return_dict=False)[0]
            
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
        reward_training:bool=False, 
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

        #print("latents",latents.size())
        #print("t",timesteps.size())
        #print("prompt_embeds",prompt_embeds.size())
        #print("image embeds",image_embeds[0].size(),image_embeds[0].device)
        
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
                #print("model pred grad",model_pred.requires_grad)
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
                #print('model_pred.device',model_pred.device,'t device',t.device,'latents',latents.device)
                t=t.to(model_pred.device)
                #latents=latents.to(model_pred.device)
                latents, denoised = self.scheduler.step(model_pred, t, latents, **extra_step_kwargs, return_dict=False)
                #print("latents grad",latents.requires_grad)
                #print("model pred grad",model_pred.requires_grad)
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
        #print("denoised grad",denoised.requires_grad)
        has_nsfw_concept = None
        if not output_type == "latent":
            image = self.vae.decode(denoised / self.vae.config.scaling_factor, return_dict=False)[0]
            do_denormalize = [True] * image.shape[0]
            try:
                image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
            except RuntimeError:
                image = self.image_processor.postprocess(image.detach(), output_type=output_type, do_denormalize=do_denormalize)
        else:
            image = denoised

        
        

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
        text_encoder_type=str(type(self.text_encoder))
        if text_encoder_type.find("DistributedDataParallel")!=-1:
            positive,negative=encode_prompt_distributed(self,prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds, negative_prompt_embeds, lora_scale, clip_skip)
        else:
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
        

class CompatibleSanaSprintPipeline(SanaSprintPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 2,
        timesteps: List[int] = None,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float = 1.3,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: int = 1024,
        width: int = 1024,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: List[str] = [
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ],
    ) -> Union[SanaPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            max_timesteps (`float`, *optional*, defaults to 1.57080):
                The maximum timestep value used in the SCM scheduler.
            intermediate_timesteps (`float`, *optional*, defaults to 1.3):
                The intermediate timestep value used in SCM scheduler (only used when num_inference_steps=2).
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://huggingface.co/papers/2010.02502. Only
                applies to [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            attention_kwargs:
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to `300`):
                Maximum sequence length to use with the `prompt`.
            complex_human_instruction (`List[str]`, *optional*):
                Instructions for complex human attention:
                https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.

        Examples:

        Returns:
            [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.sana.pipeline_output.SanaPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = self.attention_kwargs.get("scale", None) if self.attention_kwargs is not None else None

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=None,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        latents = latents * self.scheduler.config.sigma_data

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(prompt_embeds.dtype)
        guidance = guidance * self.transformer.config.guidance_embeds_scale

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        timesteps = timesteps[:-1]
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        transformer_dtype = self.transformer.dtype
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                latents_model_input = latents / self.scheduler.config.sigma_data

                scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))

                scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)
                latent_model_input = latents_model_input * torch.sqrt(
                    scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
                )

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]

                noise_pred = (
                    (1 - 2 * scm_timestep_expanded) * latent_model_input
                    + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2) * noise_pred
                ) / torch.sqrt(scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2)
                noise_pred = noise_pred.float() * self.scheduler.config.sigma_data

                # compute previous image: x_t -> x_t-1
                latents, denoised = self.scheduler.step(
                    noise_pred, timestep, latents, **extra_step_kwargs, return_dict=False
                )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = denoised / self.scheduler.config.sigma_data
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            try:
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            except torch.cuda.OutOfMemoryError as e:
                warnings.warn(
                    f"{e}. \n"
                    f"Try to use VAE tiling for large images. For example: \n"
                    f"pipe.vae.enable_tiling(tile_sample_min_width=512, tile_sample_min_height=512)"
                )
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)