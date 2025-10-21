# Imports Required:
import os
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Load data set
def data_loader(dir, batch_size=64, split=0.3, seed=3710):
    # augement data

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    dataset = datasets.ImageFolder(dir, transform=transform)

    train_size = int(len(dataset) * (1 - split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True) 
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)    

    return train_dataloader, val_dataloader


def test_loader(dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)

    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return val_dataloader





