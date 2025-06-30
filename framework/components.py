""" Components of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out * x
        return out



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,freeze_state=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.freeze_state = freeze_state
        if self.freeze_state:
            freeze(self.maxpool_conv)


    def forward(self, x):
        x = self.maxpool_conv(x)
        b, c, w, h = x.shape

        # if self.freeze_state:
        #     if train:
        #         x = x.reshape(2, -1, c, w * h).permute(0, 2, 1, 3)
        #         x = self.conv1(x)
        #         x = x.permute(0, 2, 1, 3)
        #         x = x.reshape(b, c, w, h)
        #     else:
        #         x = x.reshape(1, -1, c, w * h).permute(0, 2, 1, 3)
        #         x = self.conv1(x)
        #         x = x.permute(0, 2, 1, 3)
        #         x = x.reshape(b, c, w, h)
        #     return x
        # else:
        #     return x


        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,freeze_state=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.freeze_state = freeze_state
        if freeze_state:
            freeze(self.conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        b, c, w, h = x.shape

        # if self.freeze_state:
        #     if train:
        #         x = x.reshape(2, -1, c, w * h).permute(0, 2, 1, 3)
        #         x = self.conv1(x)
        #         x = x.permute(0, 2, 1, 3)
        #         x = x.reshape(b, c, w, h)
        #     else:
        #         x = x.reshape(1, -1, c, w * h).permute(0, 2, 1, 3)
        #         x = self.conv1(x)
        #         x = x.permute(0, 2, 1, 3)
        #         x = x.reshape(b, c, w, h)
        #     return x
        # else:
        #     return x
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))


class G_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G_OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))