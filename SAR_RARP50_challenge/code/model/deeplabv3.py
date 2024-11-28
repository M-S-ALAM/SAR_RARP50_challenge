import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv_1x1_1(x)))
        x2 = self.relu(self.bn2(self.conv_3x3_1(x)))
        x3 = self.relu(self.bn3(self.conv_3x3_2(x)))
        x4 = self.relu(self.bn4(self.conv_3x3_3(x)))
        x5 = self.avg_pool(x)
        x5 = F.interpolate(self.relu(self.bn5(self.conv_1x1_2(x5))), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.project(x)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, 256)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.relu(self.bn(self.conv(x)))
        x = self.final_conv(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.backbone_features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.classifier = DeepLabHead(2048, num_classes)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, x):
        features = self.backbone_features(x)
        output = self.classifier(features)
        output = self.upsample(output)
        return output


# Usage example
model = DeepLabV3Plus(num_classes=10)  # Adjusted to 10 classes as per your example
input_tensor = torch.rand((1, 3, 512, 512))  # Example input tensor
output = model(input_tensor)
print(output.shape)  # Expected shape: (1, 10, 512, 512)
