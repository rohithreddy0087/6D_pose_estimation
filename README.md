# 6D Pose Estimation

This project implements a sophisticated 6D pose estimation system using a combination of DeepLabv3 for object segmentation and a hybrid approach of Iterative Closest Point (ICP) and Dense Fusion for pose estimation. The system is designed to predict the 6D poses of objects in synthetic scenes, utilizing RGB images and depth maps.

## Introduction

6D object pose estimation plays a vital role in various fields like robotic manipulation, augmented reality, and autonomous navigation. The challenge in this project is to accurately determine both the position and orientation of objects in a 3D space using 2D data sources.

## Methodology

### DeepLabv3 for Object Segmentation

- Utilizes DeepLabv3 for accurate segmentation of objects in synthetic scenes.
- Employs a modified convolutional neural network, often based on ResNet, optimized for segmentation tasks.
- The segmentation model is trained with softmax cross-entropy loss, handling multi-class segmentation effectively.

### Dense Fusion for Pose Estimation

- Dense Fusion is used to predict rotation and translation vectors from RGB images, depth maps, and model point clouds.
- Features from RGB and depth data are extracted and fused, allowing the network to understand object poses in 3D space.
- The iterative refinement stage typically associated with Dense Fusion is omitted in this implementation.

### Combining ICP with Dense Fusion

- The initial pose estimate from Dense Fusion is refined using the ICP algorithm.
- ICP algorithm enhances the accuracy of the pose estimation by aligning geometric features.

## Experiments

### Dataset

- The project uses a synthetic dataset comprising 79 different object classes.
- The dataset includes a split between training, validation, and testing sets with RGB images, depth maps, and segmented object labels.

### Data Preprocessing

- Involves extracting and transforming object point clouds from RGB scenes using depth maps and segmented labels.
- Both target and source point clouds are sampled to 1000 points each for consistency in processing.

### Training

- DeepLabv3 and Dense Fusion models are trained on a single GPU setup with a batch size of 8.
- Adam optimizer is used with a learning rate of \(1 \times 10^{-4}\), adjusted during the training process.
- The Dense Fusion model undergoes 1000 epochs of training.

## Results

- The system demonstrates high accuracy in both segmentation and pose estimation tasks.
- Detailed results and visualizations of segmentation and pose estimation on test images are as follows.

![image](https://github.com/rohithreddy0087/6D_pose_estimation/assets/51110057/41269275-5e5a-4cd5-b0d8-91bed5d65a66)
![image](https://github.com/rohithreddy0087/6D_pose_estimation/assets/51110057/d0e95236-1985-4a4c-ac8d-814c0aaa230b)

## Conclusion

The project successfully integrates DeepLabv3 and a hybrid ICP+DenseFusion approach for effective segmentation and pose estimation in synthetic scenes. The comprehensive training and ablation studies contribute to the robustness and precision of the system, making it a promising solution for 6D pose estimation tasks.

---
