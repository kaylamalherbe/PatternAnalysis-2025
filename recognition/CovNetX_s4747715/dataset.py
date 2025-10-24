# Imports Required:
import os
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

IMAGE_DIM = 224

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


def calc_normalization_values(dir):
  # define custom transform function
  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()
  ])
  img_tr = datasets.ImageFolder(dir, transform=transform)[0][0]
  # transform the pIL image to tensor
  # image
  # img_tr = transform(img)

  # Convert tensor image to numpy array
  img_np = np.array(img_tr)

  mean, std = img_tr.mean([1,2]), img_tr.std([1,2])

  # print mean and std
  print("mean and std before normalize:")
  print("Mean of the image:", mean)
  print("Std of the image:", std)

  return mean, std



# Load data set
def data_loader_aug(dir, batch_size=64, split=0.3, seed=3710, mean=[0.1164], std=[0.2307]):
    # augement data

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    transform = transforms.Compose([
      transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        # transforms.RandomResizedCrop(size=IMAGE_DIM, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)
    #random choose half the dataset
    # dataset, _ = random_split(dataset, [len(dataset)//4, 3*len(dataset)//4])
    # dataset, _ = random_split(dataset, [len(dataset)//4, 3*len(dataset)//4])

    train_size = int(len(dataset) * (1 - split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    print(len(train_set))
    print(len(val_set))
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader


def test_loader_aug(dir, batch_size, mean, std):
    transform = transforms.Compose([
      transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        # transforms.RandomResizedCrop(size=IMAGE_DIM, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
    ])


    dataset = datasets.ImageFolder(dir, transform=transform)

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return test_dataloader



