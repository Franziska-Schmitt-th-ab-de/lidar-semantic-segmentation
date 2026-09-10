import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_800mf


class RegNetEncoder(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super().__init__()
        self.backbone = regnet_y_800mf(pretrained=False)
        old_conv = self.backbone.stem[0]

        # Integrate 4 channel input
        self.backbone.stem[0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.backbone.fc = nn.Identity()

        self.stem = self.backbone.stem
        self.stage1 = self.backbone.trunk_output[0]
        self.stage2 = self.backbone.trunk_output[1]
        self.stage3 = self.backbone.trunk_output[2]
        self.stage4 = self.backbone.trunk_output[3]

        if pretrained:
            official_sd = regnet_y_800mf(pretrained=True).state_dict()
            new_sd = self.state_dict()
            for name, param in official_sd.items():
                if name == "stem.0.weight":
                    # Expand from (32,3,...) to (32,4,...)
                    c_out, c_in, kh, kw = param.shape
                    expanded = torch.zeros((c_out, 4, kh, kw))
                    expanded[:, :3, :, :] = param
                    # For the 4th channel, zero-init or replicate one channel:
                    expanded[:, 3:4, :, :] = param[
                        :, 2:3, :, :
                    ]  # replicate 'B' channel
                    new_sd[name] = expanded
                else:
                    new_sd[name] = param
            self.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=256, num_classes=21):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
                for in_ch in in_channels_list
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, features):
        x1, x2, x3, x4 = features
        lat4 = self.lateral_convs[3](x4)
        lat3 = self.lateral_convs[2](x3)
        lat2 = self.lateral_convs[1](x2)
        lat1 = self.lateral_convs[0](x1)

        out4 = lat4
        out3 = F.interpolate(out4, size=lat3.shape[2:], mode="nearest") + lat3
        out3 = self.fpn_convs[2](out3)

        out2 = F.interpolate(out3, size=lat2.shape[2:], mode="nearest") + lat2
        out2 = self.fpn_convs[1](out2)

        out1 = F.interpolate(out2, size=lat1.shape[2:], mode="nearest") + lat1
        out1 = self.fpn_convs[0](out1)

        seg_map = self.seg_head(out1)
        return seg_map


class RegNetSegModel(nn.Module):
    def __init__(self, num_classes=21, in_channels=4, pretrained=True):
        super().__init__()
        self.encoder = RegNetEncoder(in_channels=in_channels, pretrained=pretrained)
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 256, 256)
            x1, x2, x3, x4 = self.encoder(dummy_input)
            in_channels_list = [x1.shape[1], x2.shape[1], x3.shape[1], x4.shape[1]]

        self.decoder = Decoder(
            in_channels_list=in_channels_list, out_channels=256, num_classes=num_classes
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        seg = self.decoder([x1, x2, x3, x4])
        seg = F.interpolate(seg, size=x.shape[2:], mode="bilinear", align_corners=False)
        return seg


def get_regnet_y_800_mf(num_classes=21, in_channels=4, pretrained=True):
    return RegNetSegModel(
        num_classes=num_classes, in_channels=in_channels, pretrained=pretrained
    )
