# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 22:58:11 2025

@author: railt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlainBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = F.relu(out, inplace=True)
        return out


class Plain18(nn.Module):

    def __init__(self, num_classes=10):  
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64,  2, stride=1)  
        self.layer2 = self._make_layer(128, 2, stride=2)  
        self.layer3 = self._make_layer(256, 2, stride=2)  
        self.layer4 = self._make_layer(512, 2, stride=2)  

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * PlainBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):

        layers = []
        layers.append(PlainBlock(self.in_planes, planes, stride))
        self.in_planes = planes * PlainBlock.expansion

        for _ in range(1, blocks):
            layers.append(PlainBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
