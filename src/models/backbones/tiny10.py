'''LeNet in PyTorch.'''
import torch.nn as nn


class Tiny10(nn.Module):

    def __init__(self, n_c):
        super(Tiny10, self).__init__()
        relu = nn.ReLU

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = relu()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = relu()
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = relu()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = relu()
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = relu()
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = relu()
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = relu()
        self.conv8 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn8 = nn.BatchNorm2d(64)
        self.relu8 = relu()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_channel_num = 64

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

def tiny10(**kwargs):
    encoder = Tiny10(28, 2, **kwargs)
    return encoder