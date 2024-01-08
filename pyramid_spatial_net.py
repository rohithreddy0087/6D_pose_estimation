import torch
from torch import nn
import torch.nn.functional as F
import resnets as resnets

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.pooling_stages = nn.ModuleList([self._build_pooling_stage(in_channels, size) for size in pool_sizes])
        self.bottleneck_conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def _build_pooling_stage(self, in_channels, pool_size):
        pooling_layer = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        conv_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(pooling_layer, conv_layer)

    def forward(self, features):
        height, width = features.size(2), features.size(3)
        pooled = [F.interpolate(stage(features), size=(height, width), mode='bilinear', align_corners=False) for stage in self.pooling_stages]
        concatenated = torch.cat(pooled + [features], dim=1)
        bottleneck = self.bottleneck_conv(concatenated)
        return self.relu(bottleneck)

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleLayer, self).__init__()
        self.conv_upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv_upsample(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

class PSPNet(nn.Module):
    def __init__(self, num_classes=21, pool_sizes=(1, 2, 3, 6), psp_channels=2048, feature_size=1024, backbone='resnet18', pretrained=False):
        super(PSPNet, self).__init__()
        self.feature_extractor = getattr(resnets, backbone)(pretrained)
        self.pyramid_pooling = PyramidPoolingModule(psp_channels, feature_size, pool_sizes)
        self.dropout1 = nn.Dropout2d(p=0.3)

        self.upsample1 = UpsampleLayer(feature_size, 256)
        self.upsample2 = UpsampleLayer(256, 64)
        self.upsample3 = UpsampleLayer(64, 64)

        self.dropout2 = nn.Dropout2d(p=0.15)
        self.final_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.classification = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        features, _ = self.feature_extractor(x) 
        pyramid_features = self.pyramid_pooling(features)
        pyramid_features = self.dropout1(pyramid_features)

        upsampled_features = self.upsample1(pyramid_features)
        upsampled_features = self.dropout2(upsampled_features)
        upsampled_features = self.upsample2(upsampled_features)
        upsampled_features = self.dropout2(upsampled_features)
        upsampled_features = self.upsample3(upsampled_features)

        output = self.final_conv(upsampled_features)
        return F.log_softmax(output, dim=1), self.classification(pyramid_features.mean([2, 3]))
