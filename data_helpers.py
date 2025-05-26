import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

class CustomTripleDataset(Dataset):
    def __init__(self,image_list,embeds_list,text_list,size):
        super().__init__()
        self.image_list=image_list
        self.embeds_list=embeds_list
        self.text_list=text_list
        self.size=size
        self.trans=transforms.Compose([
            transforms.Resize((size,size)),
             transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return self.trans(self.image_list[index]),self.embeds_list[index],self.text_list[index]