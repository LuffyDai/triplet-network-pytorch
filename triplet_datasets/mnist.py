from torchvision import datasets, transforms
import os
import torch
from .base import TripletDataset as Base


class TripletDataset(Base):
    name = 'mnist'

    def __init__(self, root, train):
        self.base_dataset = \
            datasets.MNIST(root=root, train=train, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        super(TripletDataset, self).__init__()
