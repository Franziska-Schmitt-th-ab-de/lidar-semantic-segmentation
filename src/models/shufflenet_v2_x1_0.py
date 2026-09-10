import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ShuffleNetV2Segmentation(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(ShuffleNetV2Segmentation, self).__init__()

        # Lade das ShuffleNetV2 Modell mit x1.0
        backbone = models.shufflenet_v2_x1_0(weights=None)

        # Anpassen der ersten Convolution-Schicht
        # Originale Eingabekanäle sind 3, wir setzen sie auf 'in_channels'
        backbone.conv1[0] = nn.Conv2d(
            in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Entfernen des Klassifikationskopfes
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Segmentierungskopf hinzufügen (Angepasste Kanäle)
        # Wir ändern die Eingabe-Kanäle auf 464, da das Modell diese Anzahl nach stage4 ausgibt
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(464, 512, kernel_size=3, padding=1),  # 464 Kanäle statt 576
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Durchlaufen des Backbones
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.stage2(x)
        x = self.backbone.stage3(x)
        x = self.backbone.stage4(x)

        # Anwenden des Segmentierungskopfs
        x = self.segmentation_head(x)

        # Upsampling auf die Eingabegröße
        x = F.interpolate(x, scale_factor=32, mode="bilinear", align_corners=False)

        return x  # Ausgabeform: [B, num_classes, H, W]


def get_shufflenet_v2_x1_0(num_classes):
    return ShuffleNetV2Segmentation(num_classes=num_classes, in_channels=4)
