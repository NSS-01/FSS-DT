"""
AlexNet for MNIST dataset
"""
import time

import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11,  stride=4, padding=9)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(96) # 5*5*96


        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(256, 256)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(256, 256)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(256, 10)
        self.relu8 = nn.ReLU()

    def forward(self, x):
        s_t = time.time()
        x = self.conv1(x)
        e_t = time.time()
        print("conv1 :", e_t - s_t)

        s_t = time.time()
        x = self.pool1(x)
        e_t = time.time()
        print("pool1 :", e_t - s_t)

        s_t = time.time()
        x = self.relu1(x)
        e_t = time.time()
        print("relu1 :", e_t - s_t)

        s_t = time.time()
        x = self.bn1(x)
        e_t = time.time()
        print("bn1 :", e_t - s_t)

        s_t = time.time()

        x = self.conv2(x)
        e_t = time.time()
        print("conv2 :", e_t - s_t)

        s_t = time.time()
        x = self.pool2(x)
        e_t = time.time()
        print("pool2 :", e_t - s_t)

        s_t = time.time()
        x = self.relu2(x)
        e_t = time.time()
        print("relu2 :", e_t - s_t)

        s_t = time.time()
        x = self.bn2(x)
        e_t = time.time()
        print("bn2 :", e_t - s_t)

        s_t = time.time()
        x = self.conv3(x)
        e_t = time.time()
        print("conv3 :", e_t - s_t)

        s_t = time.time()
        x = self.relu3(x)
        e_t = time.time()
        print("relu3 :", e_t - s_t)

        s_t = time.time()
        x = self.conv4(x)
        e_t = time.time()
        print("conv4 :", e_t - s_t)

        s_t = time.time()
        x = self.relu4(x)
        e_t = time.time()
        print("relu4 :", e_t - s_t)

        s_t = time.time()
        x = self.conv5(x)
        e_t = time.time()
        print("conv5 :", e_t - s_t)

        s_t = time.time()
        x = self.relu5(x)
        e_t = time.time()
        print("relu5 :", e_t - s_t)

        s_t = time.time()

        x = x.view(-1, 256)
        x = self.fc6(x)
        e_t = time.time()
        print("fc6 :", e_t - s_t)

        s_t = time.time()
        x = self.relu6(x)
        e_t = time.time()
        print("relu6 :", e_t - s_t)

        s_t = time.time()
        x = self.fc7(x)
        e_t = time.time()
        print("fc7 :", e_t - s_t)

        s_t = time.time()
        x = self.relu7(x)
        e_t = time.time()
        print("relu7 :", e_t - s_t)

        s_t = time.time()
        x = self.fc8(x)
        e_t = time.time()
        print("fc8 :", e_t - s_t)

        s_t = time.time()
        x = self.relu8(x)
        e_t = time.time()
        print("relu8 :", e_t - s_t)

        return x
