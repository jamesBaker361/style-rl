import torch
from diffusers import StableDiffusionPipeline,AutoPipelineForText2Image
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel

class SpecialPipe(AutoPipelineForText2Image):
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        print("called prepare_ip_adapter_image_embeds ")
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):

                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                print(single_image_embeds.size())

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
        print("len",len(single_image_embeds))
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)
                single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            print("single",single_image_embeds.size())
            ip_adapter_image_embeds.append(single_image_embeds)
        print("len ip",len(ip_adapter_image_embeds))
        return ip_adapter_image_embeds
    

from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

pipeline = SpecialPipe.from_pretrained("CompVis/stable-diffusion-v1-4")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality, wearing sunglasses',
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=1,
    generator=generator,
).images[0]
