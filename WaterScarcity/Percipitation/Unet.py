import torch
import torch.nn as nn
import torch.nn.functional as F

# Cropping helper
def center_crop(enc_feat, target_feat):
    _, _, h, w = enc_feat.size()
    _, _, ht, wt = target_feat.size()
    x1 = (h - ht) // 2
    y1 = (w - wt) // 2
    return enc_feat[:, :, x1:x1+ht, y1:y1+wt]

# Double conv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Full U-Net
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.enc1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x4_cropped = center_crop(x4, x)
        x = self.dec4(torch.cat([x, x4_cropped], dim=1))

        x = self.up3(x)
        x3_cropped = center_crop(x3, x)
        x = self.dec3(torch.cat([x, x3_cropped], dim=1))

        x = self.up2(x)
        x2_cropped = center_crop(x2, x)
        x = self.dec2(torch.cat([x, x2_cropped], dim=1))

        x = self.up1(x)
        x1_cropped = center_crop(x1, x)
        x = self.dec1(torch.cat([x, x1_cropped], dim=1))

        return self.final_conv(x)
