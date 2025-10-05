import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.fc = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.fc(z)       
        return x
    

class Resnet18_GN(nn.Module):
    """
    ResNet-18 with GroupNorm (32 groups) everywhere, CIFAR-friendly stem.
    - From scratch (no pretrained weights).
    - conv1: 3x3, stride=1, padding=1
    - maxpool removed
    - fc: 512 -> num_classes
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Use a factory so ResNet can call norm_layer(num_channels)
        norm_layer = lambda c: nn.GroupNorm(32, c)

        # Build resnet18 with our norm layer
        self.model = models.resnet18(weights=None, norm_layer=norm_layer)

        # CIFAR stem
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

        # Replace classifier head
        in_feats = self.model.fc.in_features  # 512 for resnet18
        self.model.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.model(x)

