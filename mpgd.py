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
    target=None,
    embedding_model: EmbeddingUtil=None,
    **kwargs,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
            The width in pixels of the generated image.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.5):
            A higher guidance scale value encourages the model to generate images closely linked to the text
            `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide what to not include in image generation. If not defined, you need to
            pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
            to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
            not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
        ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
            Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
            IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
            contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
            provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
            plain tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
            [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
            using zero terminal SNR.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
            otherwise a `tuple` is returned where the first element is a list with the generated images and the
            second element is a list of `bool`s indicating whether the corresponding generated image contains
            "not-safe-for-work" (nsfw) content.
    """
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

                new_latents=self.scheduler.add_noise(new_denoised,noise_pred,t)
                latents, denoised = self.scheduler.step(noise_pred, t, new_latents, **extra_step_kwargs, return_dict=False)

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

    return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


if __name__=="__main__":
    ddim = DDIMScheduler.from_config("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")
    pipeline=StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",torch_dtype=torch.float16,
                                                     #force_download=True,
                                                     scheduler=ddim).to("cuda")
    #pipeline.scheduler=CompatibleDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.vae.requires_grad_(False)
    dim=128
    target_image=load_image("https://media.vogue.fr/photos/5c8a55363d44a0083ccbef54/2:3/w_2560%2Cc_limit/GettyImages-625257378.jpg")
    target_tensor=pipeline.image_processor.preprocess(target_image,dim,dim).to("cuda",dtype=torch.float16,)

    embedding_model=EmbeddingUtil(pipeline.unet.device,pipeline.unet.dtype, "clip","key",4)
    target=embedding_model.embed_img_tensor(target_tensor)
    print('target size',target.size())

    for steps in [20,40]:

        
        generator=torch.Generator(pipeline.unet.device)
        generator.manual_seed(123)
        grad_image=ddim_call_with_guidance(pipeline,"cat",dim,dim,target=target,generator=generator,num_inference_steps=steps,embedding_model=embedding_model).images[0]
        

        generator=torch.Generator(pipeline.unet.device)
        generator.manual_seed(123)
        image=ddim_call_with_guidance(pipeline,"cat",dim,dim,generator=generator,num_inference_steps=steps).images[0]

        generator=torch.Generator(pipeline.unet.device)
        generator.manual_seed(123)
        normal_image=pipeline("cat",dim,dim,generator=generator,num_inference_steps=steps).images[0]
        
        concat_image=concat_images_horizontally([normal_image,image,grad_image])

        concat_image.save(f"concat_{steps}.png")

        print(f"all done {steps} ")