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
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(self.conv4(x), training=self.training)
        return x

class STL10Net(nn.Module):
    def __init__(self):
        super(STL10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(self.conv4(x), training=self.training)
        return x

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.dropout(F.relu(x), training=self.training)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x

