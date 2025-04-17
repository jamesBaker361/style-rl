from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline,AlignPropConfig,AlignPropTrainer
import torch
from collections import defaultdict

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