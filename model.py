from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyramid_spatial_net import PSPNet

class PSPModels:
    def __init__(self):
        self.models = {
            'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
            'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
            'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
            'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
            'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
        }

    def get(self, model_name):
        return self.models[model_name.lower()]()

class ModifiedResnet(nn.Module):
    def __init__(self, model='resnet18'):
        super(ModifiedResnet, self).__init__()
        self.network = PSPModels().get(model)
        self.network = nn.DataParallel(self.network)

    def forward(self, x):
        return self.network(x)

class PoseNetFeatureExtractor(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeatureExtractor, self).__init__()
        self.point_cloud_conv1 = nn.Conv1d(3, 64, 1)
        self.embedded_feat_conv1 = nn.Conv1d(32, 64, 1)
        self.point_cloud_conv2 = nn.Conv1d(64, 128, 1)
        self.embedded_feat_conv2 = nn.Conv1d(64, 128, 1)

        self.combined_feat_conv1 = nn.Conv1d(256, 512, 1)
        self.combined_feat_conv2 = nn.Conv1d(512, 1024, 1)

        self.global_feat_pool = nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, point_cloud, embedded_features):
        point_feat1 = F.relu(self.point_cloud_conv1(point_cloud))
        emb_feat1 = F.relu(self.embedded_feat_conv1(embedded_features))
        combined_feat1 = torch.cat((point_feat1, emb_feat1), dim=1)

        point_feat2 = F.relu(self.point_cloud_conv2(point_feat1))
        emb_feat2 = F.relu(self.embedded_feat_conv2(emb_feat1))
        combined_feat2 = torch.cat((point_feat2, emb_feat2), dim=1)

        combined_feat3 = F.relu(self.combined_feat_conv1(combined_feat2))
        combined_feat3 = F.relu(self.combined_feat_conv2(combined_feat3))

        global_feat = self.global_feat_pool(combined_feat3)
        global_feat = global_feat.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        
        return torch.cat([combined_feat1, combined_feat2, global_feat], 1) 

class PoseNet(nn.Module):
    def __init__(self, num_points, num_objects):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feature_extractor = PoseNetFeatureExtractor(num_points)

        self.pose_estimation_conv_layers = nn.ModuleList([
            nn.Conv1d(1408, 640, 1),
            nn.Conv1d(640, 256, 1),
            nn.Conv1d(256, 128, 1)
        ])
        
        self.rotation_output = nn.Conv1d(128, num_objects*4, 1)
        self.translation_output = nn.Conv1d(128, num_objects*3, 1)
        self.confidence_output = nn.Conv1d(128, num_objects*1, 1)

    def forward(self, image, point_cloud, choose, object_indices):
        image_features = self.cnn(image)
        bs, di, _, _ = image_features.size()
        image_features = image_features.view(bs, di, -1)
        
        choose = choose.repeat(1, di, 1)
        selected_features = torch.gather(image_features, 2, choose).contiguous()

        point_cloud = point_cloud.transpose(2, 1).contiguous()
        point_cloud_features = self.feature_extractor(point_cloud, selected_features)

        rotation_features = point_cloud_features
        translation_features = point_cloud_features
        confidence_features = point_cloud_features

        for conv in self.pose_estimation_conv_layers:
            rotation_features = F.relu(conv(rotation_features))
            translation_features = F.relu(conv(translation_features))
            confidence_features = F.relu(conv(confidence_features))

        rotation_output = self.rotation_output(rotation_features).view(bs, self.num_objects, 4, self.num_points)
        translation_output = self.translation_output(translation_features).view(bs, self.num_objects, 3, self.num_points)
        confidence_output = torch.sigmoid(self.confidence_output(confidence_features)).view(bs, self.num_objects, 1, self.num_points)
        
        out_rotation = torch.index_select(rotation_output, 1, object_indices).transpose(2, 1).contiguous()
        out_translation = torch.index_select(translation_output, 1, object_indices).transpose(2, 1).contiguous()
        out_confidence = torch.index_select(confidence_output, 1, object_indices).transpose(2, 1).contiguous()
        
        return out_rotation, out_translation, out_confidence, selected_features.detach()
