import torch.nn as nn
import torch

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=64):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, base_filters)
        self.enc2 = conv_block(base_filters, base_filters*2)
        self.enc3 = conv_block(base_filters*2, base_filters*4)
        self.bot  = conv_block(base_filters*4, base_filters*8)
        self.up3  = nn.ConvTranspose2d(base_filters*8, base_filters*4, 2, stride=2)
        self.dec3 = conv_block(base_filters*8, base_filters*4)
        self.up2  = nn.ConvTranspose2d(base_filters*4, base_filters*2, 2, stride=2)
        self.dec2 = conv_block(base_filters*4, base_filters*2)
        self.up1  = nn.ConvTranspose2d(base_filters*2, base_filters, 2, stride=2)
        self.dec1 = conv_block(base_filters*2, base_filters)
        self.final = nn.Conv2d(base_filters, out_channels, 1)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        b  = self.bot(p3)
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)
        return self.final(d1)
