import torch
import torchvision
import torchvision.transforms as transforms

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
train_loader = torch.utils.data.DataLoader(train_set)

