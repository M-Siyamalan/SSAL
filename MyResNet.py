from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
import torch.nn as nn
import  torch

class MyResNet(nn.Module):
    def __init__(self, arch, num_classes):
        super(MyResNet, self).__init__()
        if arch == 'resnet18':
            self.backbone = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        elif arch == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        nfea = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.dout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(nfea, num_classes)

    def forward(self, x):
        x = torch.flatten(self.backbone(x),1)
        logits = self.classifier(self.dout(x))
        return logits, x