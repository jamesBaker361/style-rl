from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial
from pipelines import *
from diffusers.utils.loading_utils import load_image

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
    #print("latents befor eloop",latents.requires_grad)
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

            guidance_scale=10.0
            if target is not None:
                with torch.enable_grad():
                    decoded=self.vae.decode(denoised.clone().detach()).sample
                    diff=torch.nn.functional.mse_loss(decoded,target)

                    diff_gradient=torch.autograd.grad(outputs=diff,inputs=decoded)[0]

                    print(diff_gradient.size(),decoded.size())

                new_denoised=self.vae.encode(decoded-diff_gradient.detach()).latent_dist.sample()

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


if __name__=="__main__":
    pipeline=CompatibleLatentConsistencyModelPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7").to("cuda")
    dim=256
    target_image=load_image("https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/377_133_3542_2834/master/3542.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=34d32522f47e4a67286f9894fc81c863")
    target=pipeline.image_processor.preprocess(target_image,dim,dim).to("cuda")

    steps=32

    print('target size',target.size())
    generator=torch.Generator(pipeline.unet.device)
    generator.manual_seed(123)
    image=call_with_grad_and_guidance(pipeline,"cat",256,256,target=target,generator=generator,num_inference_steps=steps).images[0]
    image.save("grad_cat.png")

    generator=torch.Generator(pipeline.unet.device)
    generator.manual_seed(123)
    image=call_with_grad_and_guidance(pipeline,"cat",256,256,generator=generator,num_inference_steps=steps).images[0]
    image.save("cat.png")