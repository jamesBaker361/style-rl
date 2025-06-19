from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_scm import SCMScheduler
import torch

class CompatibleSCMScheduler(SCMScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:


        t=torch.arctan(torch.exp(timesteps)/self.config.sigma_data).view(-1,1,1,1)
        noise=self.config.sigma_data*noise

        print("t size",t.size())
        print("original sample",original_samples.size())
        print("noise",noise.size())

        noisy_model_input = torch.cos(t) * original_samples + torch.sin(t) * noise
        return noisy_model_input/self.config.sigma_data,t,noise

class CompatibleFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_model_input = (1.0 - sigma) * original_samples + sigma * noise
        return noisy_model_input
    
    def get_velocity(self,latents, noise, timesteps):
        return noise-latents