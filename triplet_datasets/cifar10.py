import torch
from torchvision import datasets, transforms
import os
from .base import TripletDataset as Base


class TripletDataset(Base):
    name = 'cifar10'

    def __init__(self, root, train):
        if train:
            self.base_dataset = datasets.CIFAR10(
                root=root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            self.base_dataset = datasets.CIFAR10(
                root=root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        super(TripletDataset, self).__init__()