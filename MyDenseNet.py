from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch


class MyDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(MyDenseNet, self).__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        nfea = self.backbone.classifier.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children()))[0]

        self.dout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(nfea, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(self.dout(x)), x