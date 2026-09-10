import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.models import ShuffleNet_V2_X1_5_Weights


class ShuffleNetV2Segmentation(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(ShuffleNetV2Segmentation, self).__init__()
        # Vortrainierte Gewichte laden
        backbone = models.shufflenet_v2_x1_5(
            weights=ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1
        )

        backbone.conv1[0] = nn.Conv2d(
            in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(704, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)

        x = self.segmentation_head(x)

        # Upsampling
        x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)

        return x


def get_shufflenet_v2_x1_5(num_classes):
    return ShuffleNetV2Segmentation(num_classes=num_classes, in_channels=4)
