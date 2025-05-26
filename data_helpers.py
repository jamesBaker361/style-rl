import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomTripleDataset(Dataset):
    def __init__(self,image_list,embeds_list,text_list):
        super().__init__()
        self.image_list=image_list
        self.embeds_list=embeds_list
        self.text_list=text_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return self.image_list[index],self.embeds_list[index],self.text_list[index]