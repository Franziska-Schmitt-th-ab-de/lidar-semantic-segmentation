import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super(ResNet34Encoder, self).__init__()
        # Laden des vortrainierten ResNet34-Modells
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Anpassen der ersten Convolution-Schicht, um 'in_channels' zu akzeptieren
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool

        # Encoder-Schichten
        self.layer1 = resnet34.layer1  # Ausgabe: 64 Kanäle
        self.layer2 = resnet34.layer2  # Ausgabe: 128 Kanäle
        self.layer3 = resnet34.layer3  # Ausgabe: 256 Kanäle
        self.layer4 = resnet34.layer4  # Ausgabe: 512 Kanäle

        # Initialisierung der angepassten ersten Convolution-Schicht
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x0 = self.conv1(x)  # [B, 64, H/2, W/2]
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # [B, 64, H/2, W/2]
        x0_pool = self.maxpool(x0)  # [B, 64, H/4, W/4] # ab hier Problem??
        x1 = self.layer1(x0_pool)  # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 512, H/32, W/32]

        return x4, x3, x2, x1, x0  # Rückgabe von x0 vor dem MaxPooling


class SegmentationDecoder(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationDecoder, self).__init__()

        # Decoder-Schichten mit Upsampling und Skip-Connections
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.iconv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.iconv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(
                num_groups=16, num_channels=128
            ),  # besser bei kleinerer batch_size; vorher: nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.iconv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(
                num_groups=8, num_channels=64
            ),  # besser bei kleinerer batch_size; vorher: nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.upconv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.iconv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(
                num_groups=8, num_channels=64
            ),  # besser bei kleinerer batch_size; vorher: nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x4, x3, x2, x1, x0):
        # x4: [B, 512, H/32, W/32]
        # x3: [B, 256, H/16, W/16]
        # x2: [B, 128, H/8, W/8]
        # x1: [B, 64, H/4, W/4]
        # x0: [B, 64, H/2, W/2]

        x = self.upconv1(x4)  # [B, 256, H/16, W/16]
        x = torch.cat([x, x3], dim=1)  # [B, 512, H/16, W/16]
        x = self.iconv1(x)  # [B, 256, H/16, W/16]

        x = self.upconv2(x)  # [B, 128, H/8, W/8]
        x = torch.cat([x, x2], dim=1)  # [B, 256, H/8, W/8]
        x = self.iconv2(x)  # [B, 128, H/8, W/8]

        x = self.upconv3(x)  # [B, 64, H/4, W/4]
        x = torch.cat([x, x1], dim=1)  # [B, 128, H/4, W/4]
        x = self.iconv3(x)  # [B, 64, H/4, W/4]

        x = self.upconv4(x)  # [B, 64, H/2, W/2]
        x = torch.cat([x, x0], dim=1)  # [B, 128, H/2, W/2]
        x = self.iconv4(x)  # [B, 64, H/2, W/2]

        x = self.conv_final(x)  # [B, num_classes, H/2, W/2]

        return x


class ResNet34Segmentation(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(ResNet34Segmentation, self).__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels)
        self.decoder = SegmentationDecoder(num_classes=num_classes)

    def forward(self, x):
        x4, x3, x2, x1, x0 = self.encoder(x)
        x = self.decoder(x4, x3, x2, x1, x0)
        # Upsampling auf die Eingabegröße
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return x  # Ausgabeform: [B, num_classes, H, W]


def get_resnet34(num_classes, **kwargs):
    return ResNet34Segmentation(num_classes=num_classes, **kwargs)
