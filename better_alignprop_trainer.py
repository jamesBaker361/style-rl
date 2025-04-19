from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,AlignPropConfig,AlignPropTrainer
import torch
from collections import defaultdict
from pipelines import PPlusCompatibleLatentConsistencyModelPipeline

class BetterAlignPropTrainer(AlignPropTrainer):
    def step(self, epoch: int, global_step: int):
        """
        Perform a single step of training.

        Args:
            epoch (int): The current epoch.
            global_step (int): The current global step.

        Side Effects:
            - Model weights are updated
            - Logs the statistics to the accelerator trackers.
            - If `self.image_samples_callback` is not None, it will be called with the prompt_image_pairs, global_step, and the accelerator tracker.

        Returns:
            global_step (int): The updated global step.
        """
        info = defaultdict(list)

        self.sd_pipeline.unet.train()

        for _ in range(self.config.train_gradient_accumulation_steps):
            with self.accelerator.accumulate(self.sd_pipeline.unet), self.autocast(), torch.enable_grad():
                prompt_image_pairs = self._generate_samples(
                    batch_size=self.config.train_batch_size,
                )

                rewards = self.compute_rewards(prompt_image_pairs)

                prompt_image_pairs["rewards"] = rewards

                rewards_vis = self.accelerator.gather(rewards).detach().to(torch.float32).cpu().numpy()

                loss = self.calculate_loss(rewards)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.trainable_layers.parameters()
                        if not isinstance(self.trainable_layers, list)
                        else self.trainable_layers,
                        self.config.train_max_grad_norm,
                    )
                p_count=0
                '''for name,param in self.sd_pipeline.unet.named_parameters():
                    if param.grad is not None:
                        p_count+=1
                        grad_abs_sum = param.grad.abs().sum()
                        print(f"Sum of absolute values of gradients: {grad_abs_sum.item()}")
                print("counted p params with grad", p_count)'''
                self.optimizer.step()
                self.optimizer.zero_grad()

            info["reward_mean"].append(rewards_vis.mean())
            info["reward_std"].append(rewards_vis.std())
            info["loss"].append(loss.item())

        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # log training-related stuff
            info = {k: torch.mean(torch.tensor(v)) for k, v in info.items()}
            info = self.accelerator.reduce(info, reduction="mean")
            info.update({"epoch": epoch})
            self.accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)
        else:
            raise ValueError(
                "Optimization step should have been performed by this point. Please check calculated gradient accumulation settings."
            )
        # Logs generated images
        if self.image_samples_callback is not None:
            self.image_samples_callback(prompt_image_pairs, global_step, self.accelerator.trackers[0])

        if epoch != 0 and epoch % self.config.save_freq == 0 and self.accelerator.is_main_process:
            self.accelerator.save_state()

        return global_step
    
    def _generate_samples(self, batch_size, with_grad=True, prompts=None):
        """
        Generate samples from the model

        Args:
            batch_size (int): Batch size to use for sampling
            with_grad (bool): Whether the generated RGBs should have gradients attached to it.

        Returns:
            prompt_image_pairs (dict[Any])
        """
        prompt_image_pairs = {}

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        if prompts is None:
            prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])
        else:
            prompt_metadata = [{} for _ in range(batch_size)]

        '''if type(self.sd_pipeline)==PPlusCompatibleLatentConsistencyModelPipeline:

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)

        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]'''

        if with_grad:
            sd_output = self.sd_pipeline.rgb_with_grad(
                prompt=prompts,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                truncated_backprop_rand=self.config.truncated_backprop_rand,
                truncated_backprop_timestep=self.config.truncated_backprop_timestep,
                truncated_rand_backprop_minmax=self.config.truncated_rand_backprop_minmax,
                output_type="pt",
            )
        else:
            sd_output = self.sd_pipeline(
                prompt=prompts,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=self.config.sample_num_steps,
                guidance_scale=self.config.sample_guidance_scale,
                eta=self.config.sample_eta,
                output_type="pt",
            )

        images = sd_output.images

        prompt_image_pairs["images"] = images
        prompt_image_pairs["prompts"] = prompts
        prompt_image_pairs["prompt_metadata"] = prompt_metadata

        return prompt_image_pairs