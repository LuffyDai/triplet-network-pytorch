from torchvision import datasets, transforms
import torch
import os
from .base import TripletDataset as Base


class TripletDataset(Base):
    name = 'svhn'

    def __init__(self, root, train):
        if train:
            self.base_dataset = datasets.SVHN(
                root=root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            self.base_dataset = datasets.SVHN(
                root=root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        super(TripletDataset, self).__init__()
