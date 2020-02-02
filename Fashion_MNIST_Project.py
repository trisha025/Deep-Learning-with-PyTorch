import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)

#download data
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    
    ])
)

#data loading into an object
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10)

print("Total number of traning images: ", len(train_set))

print("Training labels: ", train_set.targets)