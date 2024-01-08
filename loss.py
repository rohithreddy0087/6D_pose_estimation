from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import numpy as np

def compute_loss(predicted_rotation, predicted_translation, predicted_confidence, 
                 true_labels, model_keypoints, batch_indices, object_points, 
                 confidence_weight, refinement_flag, mesh_point_count, symmetry_classes):
    batch_size, num_points, _ = predicted_confidence.size()
    
    predicted_rotation_norm = predicted_rotation / (torch.norm(predicted_rotation, dim=2).view(batch_size, num_points, 1))
    rotation_matrices = compute_rotation_matrix_from_quaternion(predicted_rotation_norm, batch_size, num_points)
    
    reshaped_model_keypoints = model_keypoints.view(batch_size, 1, mesh_point_count, 3).repeat(1, num_points, 1, 1).view(batch_size * num_points, mesh_point_count, 3)
    reshaped_true_labels = true_labels.view(batch_size, 1, mesh_point_count, 3).repeat(1, num_points, 1, 1).view(batch_size * num_points, mesh_point_count, 3)
    
    predicted_translation_extended = predicted_translation.contiguous().view(batch_size * num_points, 1, 3)
    object_points_extended = object_points.contiguous().view(batch_size * num_points, 1, 3)
    predicted_confidence_extended = predicted_confidence.contiguous().view(batch_size * num_points)
    
    predicted_keypoints = torch.add(torch.bmm(reshaped_model_keypoints, rotation_matrices), object_points_extended + predicted_translation_extended)
    
    distance = torch.mean(torch.norm((predicted_keypoints - reshaped_true_labels), dim=2), dim=1)
    loss = torch.mean((distance * predicted_confidence_extended - confidence_weight * torch.log(predicted_confidence_extended)), dim=0)
    
    return loss

def compute_rotation_matrix_from_quaternion(quaternion, batch_size, num_points):
    q0, q1, q2, q3 = quaternion[:, :, 0], quaternion[:, :, 1], quaternion[:, :, 2], quaternion[:, :, 3]
    r00 = 1 - 2 * (q2**2 + q3**2)
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 1 - 2 * (q1**2 + q3**2)
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 1 - 2 * (q1**2 + q2**2)
    rotation_matrix = torch.stack((r00, r01, r02, r10, r11, r12, r20, r21, r22), dim=2)
    rotation_matrices = rotation_matrix.view(batch_size * num_points, 3, 3)
    return rotation_matrices

class PoseLoss(_Loss):

    def __init__(self, num_points_mesh, symmetry_classes):
        super(PoseLoss, self).__init__(True)
        self.mesh_point_count = num_points_mesh
        self.symmetry_classes = symmetry_classes

    def forward(self, predicted_rotation, predicted_translation, predicted_confidence, true_labels, model_keypoints, batch_indices, object_points, confidence_weight, refinement_flag):
        return compute_loss(predicted_rotation, predicted_translation, predicted_confidence, true_labels, model_keypoints, batch_indices, object_points, confidence_weight, refinement_flag, self.mesh_point_count, self.symmetry_classes)
