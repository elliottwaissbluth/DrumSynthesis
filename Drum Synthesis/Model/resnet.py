import os, sys, pickle, time, librosa, torch, numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer 
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )


        
class ResNetBigger(nn.Module):
    def __init__(self, num_classes=1,dropout_rate=0.5,linear_layer_size=192):
        super(ResNetBigger, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        
        self.linear_layer_size=linear_layer_size
        
        # Create blocks
        #self.block1 = self._create_block(64, 64, stride=1)
        #self.block2 = self._create_block(64, 128, stride=2)
        #self.block3 = self._create_block(128, 128, stride=2)
        #self.block4 = self._create_block(128, 128, stride=3)
        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 32, stride=2)
        self.block3 = self._create_block(32, 16, stride=2)
        self.block4 = self._create_block(16, 16, stride=2)
        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)
      
        self.dropout = nn.Dropout(dropout_rate)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
    # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out
    
    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)