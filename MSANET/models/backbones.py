import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet101


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, arch="resnet50", pretrained=True):
        super().__init__()
        if arch == "resnet18":
            model = resnet18(weights="DEFAULT" if pretrained else None)
            self.out_channels = 512
        elif arch == "resnet50":
            model = resnet50(weights="DEFAULT" if pretrained else None)
            self.out_channels = 2048
        elif arch == "resnet101":
            model = resnet101(weights="DEFAULT" if pretrained else None)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported ResNet arch: {arch}")

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class XceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Please install timm to use xception backbone.") from e

        self.model = timm.create_model(
            "xception", pretrained=pretrained, features_only=True, out_indices=[-1]
        )
        self.out_channels = self.model.feature_info.channels()[-1]

    def forward(self, x):
        return self.model(x)[-1]


def build_backbone(name="resnet50", pretrained=True):
    name = name.lower()
    if name in ["resnet18", "resnet50", "resnet101"]:
        return ResNetFeatureExtractor(name, pretrained)
    if name == "xception":
        return XceptionFeatureExtractor(pretrained)
    raise ValueError(f"Unsupported backbone: {name}")
