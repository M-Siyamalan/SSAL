from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch


class MyMobileNet(nn.Module):
    def __init__(self, arch, num_classes):
        super(MyMobileNet, self).__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        nfea = self.backbone.classifier[-1].in_features
        self.classifier = nn.Sequential(*list(self.backbone.children()))[-1]
        self.backbone = nn.Sequential(*list(self.backbone.children()))[:-1]
        self.classifier[-1] = nn.Linear(nfea, num_classes)
        self.nfea = nfea

    def forward(self, x):
        x = torch.flatten(self.backbone(x),1)
        return self.classifier(x), x


class ShuffleNet(nn.Module):
    def __init__(self, arch, num_classes):
        super(ShuffleNet, self).__init__()
        self.backbone = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        nfea = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(nfea, num_classes)
        self.nfea = nfea

    def forward(self, x):
        x = torch.flatten(self.backbone(x),1)
        return self.classifier(x), x