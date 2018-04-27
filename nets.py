# Define networks
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class cifarANDsvhnNet(nn.Module):
    def __init__(self):
        super(cifarANDsvhnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128)
        return x

class STL10Net(nn.Module):
    def __init__(self):
        super(STL10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv4(x))
        x = x.view(-1, 128)
        return x

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, 128)
        # x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

class Classifier(nn.Module):
    def __init__(self, embedding, for_metric=True,classes=10):
        super(Classifier, self).__init__()
        self.embedding = embedding
        if not for_metric:
            for p in self.embedding.parameters():
                p.requires_grad=False
        self.fc = nn.Linear(128, classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

__all__ = ['MNISTNet', 'cifarANDsvhnNet',
           'STL10Net', 'Net',
           'Classifier']



