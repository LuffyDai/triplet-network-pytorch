from torchvision import datasets, transforms
import torch
from .base import TripletDataset as Base


class TripletDataset(Base):
    name = 'stl10'
    def __init__(self, root, train):
        if train:
            self.base_dataset = datasets.STL10(
                root=root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(96),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        else:
            self.base_dataset = datasets.STL10(
                root=root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        super(TripletDataset, self).__init__()
