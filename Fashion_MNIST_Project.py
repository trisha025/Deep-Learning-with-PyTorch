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

print("Number of images in each class: ", train_set.targets.bincount())

#to access an individual sample
sample = next(iter(train_set))
print(len(sample)) # =2 (i) image (ii) label

print(type(sample))
image, label = sample

#sample shape
print(image.shape)

plt.imshow(image.squeeze(), cmap = 'gray')
print('label: ', label)
plt.show()

#to access a batch
batch = next(iter(train_loader))

print(len(batch))
print(type(batch))
images, labels = batch

#batch shape
print(images.shape)
print(labels.shape)
