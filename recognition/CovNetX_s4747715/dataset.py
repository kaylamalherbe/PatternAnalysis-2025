# Imports Required:
import os
import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

IMAGE_DIM = 224


def calc_normalization_values(dir):
  # define custom transform function
  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()
  ])
  dataset = datasets.ImageFolder(dir, transform=transform)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

  mean = 0.0
  std = 0.0
  total_images_count = 0
  for images, _ in dataloader:
      # Rearrange batch to be the first dimension
      batch_samples = images.size(0)
      images = images.view(batch_samples, images.size(1), -1)
      mean += images.mean(2).sum(0)
      std += images.std(2).sum(0)
      total_images_count += batch_samples

  mean /= total_images_count
  std /= total_images_count

  # print mean and std
  print("mean and std before normalize:")
  print("Mean of the dataset:", mean)
  print("Std of the dataset:", std)

  return mean, std

def check_validation_split(dl_aug, val_dl_aug):
    
    # Get the underlying dataset from the DataLoaders
    # Since random_split returns Subset, we need to access the original dataset and the indices
    train_dataset = dl_aug.dataset.dataset
    train_indices = dl_aug.dataset.indices
    val_dataset = val_dl_aug.dataset.dataset
    val_indices = val_dl_aug.dataset.indices


    # Get the targets for the subsets
    train_targets = [train_dataset.targets[i] for i in train_indices]
    val_targets = [val_dataset.targets[i] for i in val_indices]


    # Count the occurrences of each class label
    train_class_counts = Counter(train_targets)
    val_class_counts = Counter(val_targets)

    print("Class distribution in dl_aug (Training DataLoader):")
    for class_idx, count in train_class_counts.items():
        # Get the class name from the dataset
        class_name = train_dataset.classes[class_idx]
        print(f"Class {class_name} ({class_idx}): {count}")

    print("\nClass distribution in val_dl_aug (Validation DataLoader):")
    for class_idx, count in val_class_counts.items():
        # Get the class name from the dataset
        class_name = val_dataset.classes[class_idx]
        print(f"Class {class_name} ({class_idx}): {count}")


def data_loader_aug(dir, batch_size=64, split=0.2, seed=3710, mean=[0.1155], std=[0.2224]):
    # augement data
    transform = transforms.Compose([
        #  transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.RandomResizedCrop((IMAGE_DIM, IMAGE_DIM), scale=(0.9, 1.0), antialias=True),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Adjust brightness and contrast slightly
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)), # Slightly increased translate and scale
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
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


def test_loader_aug(dir, batch_size, mean=[0.1155], std=[0.2224]):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return test_dataloader
