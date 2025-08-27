import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self,image_list,embeds_list,text_list,posterior_list=None,prompt_list=None,label_list=None,label_plane_list=None):
        super().__init__()
        self.image_list=image_list
        self.embeds_list=embeds_list
        self.text_list=text_list
        self.posterior_list=posterior_list
        self.prompt_list=prompt_list
        self.label_list=label_list
        self.label_plane_list=label_plane_list

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        '''print("index")
        print({
                "image":type(self.image_list[index]),
                "embeds":type(self.embeds_list[index]),
                "text":type(self.text_list[index]),
                "posterior":type(self.posterior_list[index])
            })'''
        item={
                "image":self.image_list[index],
                "embeds":self.embeds_list[index],
                "text":self.text_list[index],
            }
        if self.posterior_list is not None:
            item["posterior"]=self.posterior_list[index]
        if self.prompt_list is not None:
            item["prompt"]=self.prompt_list[index]
        if self.label_list is not None:
            item["label"]=self.label_list[index]
        if self.label_plane_list is not None:
            item["label_plane"]=self.label_plane_list[index]

        return item
        
        

class ScaleDataset(Dataset):
    def __init__(self,embedding_list,image_list,text_embedding):
        super().__init__()
        self.embedding_list=embedding_list
        self.image_list=image_list
        self.text_embedding_list=[text_embedding.clone() for _ in embedding_list]
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        return {
                "image":self.image_list[index],
                "embedding":self.embedding_list[index],
                "text_embedding":self.text_embedding_list[index]
            }
