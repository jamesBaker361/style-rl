import PIL.Image
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
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    SanaLinearAttnProcessor2_0,
    IPAdapterAttnProcessor2_0
)
import PIL
import numpy as np
from copy import deepcopy
import random
from diffusers import SanaSprintPipeline
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel, SanaTransformerBlock
from diffusers.schedulers import LCMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.sana.pipeline_output import SanaPipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN
import warnings
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.normalization import FP32LayerNorm,RMSNorm,LpNorm


def set_ip_adapter_attn(attn2:Attention,
                   device,
                   dtype,
                   ip_cross_attention_dim:int)->Attention:
    
    hidden_size=attn2.inner_kv_dim
    ip_adapter_processor=IPAdapterAttnProcessor2_0(hidden_size,ip_cross_attention_dim).to(device,dtype)
    attn2.set_processor(ip_adapter_processor)
    return attn2

def prepare_ip_adapter(model:SanaTransformer2DModel,
                   device,
                   dtype,
                   ip_cross_attention_dim:int):
    for block in model.transformer_blocks:
        block.attn2=set_ip_adapter_attn(block.attn2,device,dtype,ip_cross_attention_dim)



def compatible_forward_sana_transformer_block(
        self:SanaTransformerBlock,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        
        height: int = None,
        width: int = None,
        added_cond_kwargs:dict=None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output

        '''if added_cond_kwargs!=None:
            ip_hidden_states=added_cond_kwargs["image_embeds"]
            encoder_hidden_states=(encoder_hidden_states,ip_hidden_states)
        else:'''
        encoder_hidden_states=encoder_hidden_states

        # 3. Cross Attention
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states

def compatible_process_hidden_states(
        encoder_hid_proj:torch.nn.Module, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
    if encoder_hid_proj is not None:
        if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{encoder_hid_proj.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
        image_embeds = added_cond_kwargs.get("image_embeds")
        image_embeds = encoder_hid_proj(image_embeds)
        encoder_hidden_states = (encoder_hidden_states, image_embeds)

    return encoder_hidden_states


def compatible_forward_sana_transformer_model(
    self:SanaTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = {},
    controlnet_block_samples: Optional[Tuple[torch.Tensor]] = None,
    return_dict: bool = True,
    encoder_hid_proj:torch.nn.Module=None,
) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)


    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
    #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
    #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None and attention_mask.ndim == 2:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 1. Input
    batch_size, num_channels, height, width = hidden_states.shape
    p = self.config.patch_size
    post_patch_height, post_patch_width = height // p, width // p

    hidden_states = self.patch_embed(hidden_states)

    if guidance is not None:
        timestep, embedded_timestep = self.time_embed(
            timestep, guidance=guidance, hidden_dtype=hidden_states.dtype
        )
    else:
        timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

    encoder_hidden_states = self.caption_norm(encoder_hidden_states)

    encoder_hidden_states=compatible_process_hidden_states(encoder_hid_proj,encoder_hidden_states,added_cond_kwargs)

    # 2. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = self._gradient_checkpointing_func(
                compatible_forward_sana_transformer_block,
                block,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
                added_cond_kwargs
            )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

    else:
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = compatible_forward_sana_transformer_block(
                block,
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_height,
                post_patch_width,
                added_cond_kwargs
            )
            if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

    # 3. Normalization
    hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)

    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_height, post_patch_width, self.config.patch_size, self.config.patch_size, -1
    )
    hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
    output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


class CompatibleSanaSprintPipeline(SanaSprintPipeline):

    def set_encoder_hid_proj(self,encoder_hid_proj):
        self.encoder_hid_proj=encoder_hid_proj
        self.register_modules(encoder_hid_proj=encoder_hid_proj)
        self.register_to_config(encoder_hid_proj=encoder_hid_proj)

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            '''if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.transformer.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.transformer.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])'''
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
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Union[SanaPipelineOutput, Tuple]:
        
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

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                False,
            )
            added_cond_kwargs={"image_embeds": image_embeds}
        else:
            added_cond_kwargs=None

        # 7. Denoising loop
        timesteps = timesteps[:-1]
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        encoder_hid_proj=getattr(self,"encoder_hid_proj",None)
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
                noise_pred = compatible_forward_sana_transformer_model(
                    self.transformer,
                    latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance,
                    timestep=scm_timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    encoder_hid_proj=encoder_hid_proj
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