""" dtnet.py
    Recurrent Networks for IQ Tests
    April 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class ConvModule(nn.Module):
    def __init__(self, width):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(width, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(x)
        # x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(x)
        # x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(x)
        # x = self.relu4(self.batch_norm4(x))
        return x  # x.view(-1, 32 * 4 * 4)


class MLPModule(nn.Module):
    def __init__(self):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(32 * 4 * 4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block class"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                                                    stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DTNet(nn.Module):
    """Deep Thinking Netowrk"""

    def __init__(self, block, num_blocks, width, iters, in_channels=3, recall=True, **kwargs):
        super(DTNet, self).__init__()

        self.recall = recall
        self.width = int(width)
        self.iters = iters
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        conv_recall = nn.Conv2d(width * 2, width, kernel_size=3, stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 128, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        linear = nn.Linear(512 * 4 * 4, 8)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.AvgPool2d(2), nn.ReLU(),  # x32
                                  head_conv2, nn.AvgPool2d(2), nn.ReLU(),  # x16
                                  head_conv3, nn.AvgPool2d(4), nn.ReLU(),  # x4
                                  nn.Flatten(), linear
                                  )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, interim_thought=None, **kwargs):
        x = F.interpolate(x, size=(64, 64))
        # iters = kwargs["k"]
        iters = self.iters
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters, 8)).to(x.device)

        for i in range(iters):
            if self.recall:
                interim_thought = torch.cat([interim_thought, initial_thought], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)

            all_outputs[:, i] = out

        # if self.training:
        #     return out, interim_thought
        # else:
        #     return all_outputs

        return out, None, None


class DTNetIQ(nn.Module):
    """Modified ResidualNetworkSegment model class"""

    def __init__(self, block, num_blocks, width, iters, in_channels=3, recall=True, **kwargs):
        super(DTNetIQ, self).__init__()

        self.recall = recall
        self.width = int(width)
        self.iters = iters
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        conv_recall = nn.Conv2d(width * 2, width, kernel_size=3, stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        conv = ConvModule(16)
        flatten = nn.Flatten()
        mlp = MLPModule()

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(conv, flatten, mlp)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, interim_thought=None, **kwargs):
        x = F.interpolate(x, size=(80, 80))
        # iters = kwargs["k"]
        iters = self.iters
        all_outputs = torch.zeros((x.size(0), iters, 8)).to(x.device)
        for i in range(iters):
            out = self.head(x)
            all_outputs[:, i] = out

        # if self.training:
        #     return out, interim_thought
        # else:
        #     return all_outputs

        return out, None, None


def dt_net(width, iters, **kwargs):
    return DTNet(BasicBlock, [2], width=width, iters=iters, in_channels=16, recall=False)


def dt_net_recall(width, iters, **kwargs):
    return DTNet(BasicBlock, [2], width=width, iters=iters, in_channels=16, recall=True)


def dt_net_iq(width, iters, **kwargs):
    return DTNetIQ(BasicBlock, [2], width=width, iters=iters, in_channels=16, recall=True)
