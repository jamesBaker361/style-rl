import torch
from torch import nn

class DeeperImageProjection(nn.Module):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.out_dim=self.num_image_text_embeds * cross_attention_dim
        self.mid_dim=(num_image_text_embeds+self.out_dim)//2
        self.image_embeds = nn.Sequential(
            nn.Linear(image_embed_dim, self.mid_dim),
            nn.SiLU(),
            nn.Linear(self.mid_dim,self.out_dim),
            nn.SiLU()
        )

    def forward(self, image_embeds: torch.Tensor):
        batch_size = image_embeds.shape[0]

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        return image_embeds
    

class PromptImageProjection(DeeperImageProjection):
    def forward(self,image_embeds: torch.Tensor,positive:torch.Tensor)->torch.Tensor:
        batch_size = image_embeds.shape[0]

        #print("image embeds shape",image_embeds.size())

        # image
        image_embeds = self.image_embeds(image_embeds.to(self.image_embeds[0].weight.dtype))
        image_embeds = image_embeds.reshape(batch_size, self.num_image_text_embeds, -1)
        
        positive=positive[:, : -self.num_image_text_embeds, :]  #len 77 -> 76 this probably is just padding anyway
        #print("src embeds, positive shapes", src_embeds.size(),positive.size())
        positive=torch.cat([image_embeds,positive],dim=1)

        return positive