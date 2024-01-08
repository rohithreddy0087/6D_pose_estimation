import os
import csv
import numpy as np
import numpy.ma as ma
import pickle
from matplotlib.cm import get_cmap
from PIL import Image
import random
from collada import Collada

import torch
from torchvision import transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

NUM_OBJECTS = 79

def custom_DeepLabv3(out_channel):
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier = DeepLabHead(2048, out_channel)
    return model

class PoseDataset():
    def __init__(self, data_folder, split_dir, objects_csv, mode = "train"):
        self.symmetry_obj_idx = []
        self.rgb_files, self.depth_files, self.label_files, self.meta_files, self.prefixes = self.get_split_files(data_folder, split_dir, mode)
        self.cld = self.read_models(objects_csv)
        self.border_list = [-1]+list(np.arange(40,1320,40))
        self.img_width = 720
        self.img_length = 1280
        self.num_pt = 1000
        self.minimum_num_pt = 50
        self.num_pt_mesh_small = 500
        self.mode = mode

        seg_checkpoint_path = "/root/data/rohith/gnn/HW3/checkpoints/deeplabv3_best.pth"
        self.seg_model = custom_DeepLabv3(out_channel = NUM_OBJECTS+3).cuda()
        self.seg_model = self.seg_model.cuda()
        self.seg_model.load_state_dict(torch.load(seg_checkpoint_path))
        self.seg_model.eval()

        transform_list = [transforms.ToTensor(),
                            transforms.Resize(size=(256, 256)),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform_rgb = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.rgb_files)

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def get_split_files(self, training_data_dir, split_dir, split_name):
        with open(os.path.join(split_dir, f"{split_name}.txt"), 'r') as f:
            prefix = [os.path.join(training_data_dir, line.strip()) for line in f if line.strip()]
            rgb = [p + "_color_kinect.png" for p in prefix]
            depth = [p + "_depth_kinect.png" for p in prefix]
            label = [p + "_label_kinect.png" for p in prefix]
            meta = [p + "_meta.pkl" for p in prefix]
            prefixes = [p.split("/")[-1] for p in prefix]
        return rgb, depth, label, meta, prefixes
    
    def read_models(self, csv_file_path):
        data_dict = {}
        with open(csv_file_path, mode='r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            i = 0
            for row in csv_reader:
                data_dict[i] =  self.load_point_cloud(row['location'])
                if "symmetric" in row['metric']:
                    self.symmetry_obj_idx.append(i)
                i += 1
        print("Finished Loading")
        return data_dict
    
    def load_point_cloud(self, dae_file_path):
        collada_mesh = Collada('/root/data/rohith/gnn/HW2/'+dae_file_path+'/visual_meshes/visual.dae')
        geometry = collada_mesh.geometries[0]
        primitive = geometry.primitives[0]  
        vertices = np.array(primitive.vertex)
        return vertices

    def getitem(self, i, idx):
        try:
            rgb = np.array(Image.open(self.rgb_files[i]))/255.0 
            depth = np.array(Image.open(self.depth_files[i]))/1000.0
            label = self.seg_model(self.transform_rgb(rgb.astype(np.float32)).unsqueeze(0).cuda())['out'][0].argmax(0).cpu().numpy()
            meta = self.load_pickle(self.meta_files[i])
            prefix = self.prefixes[i]
        
            obj = meta['object_ids'].flatten().astype(np.int32)
           
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            
            rmin, rmax, cmin, cmax = self.get_bbox(mask_label)
            img_masked = np.transpose(rgb[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > self.num_pt:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:self.num_pt] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            self.xmap, self.ymap = np.indices(depth.shape)
            xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])
            
            cam_scale = meta['scales'][obj[idx]][0]
            intrinsic_matrix = meta['intrinsic']
            pt2 = depth_masked
            pt0 = (ymap_masked - intrinsic_matrix[0, 2]) * pt2 / intrinsic_matrix[0, 0]
            pt1 = (xmap_masked - intrinsic_matrix[1, 2]) * pt2 / intrinsic_matrix[1, 1]
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            ext_inv = np.linalg.inv(meta['extrinsic'])
            centroid = np.mean(cloud, axis=0)
            cloud = (cloud-centroid)/cam_scale+centroid
            cloud = cloud@ext_inv[:3,:3].T+ext_inv[:3,3]

            try:
                dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
                dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
                model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)
            except:
                model_points = self.cld[obj[idx]]

            if self.mode != "test":
                target_r = meta['poses_world'][obj[idx]][:, 0:3]
                target_t = np.array([meta['poses_world'][obj[idx]][:, 3:4].flatten()])
                target = np.dot(model_points, target_r.T)
                target = np.add(target, target_t)
                target = target[:,:-1]
                return torch.from_numpy(cloud.astype(np.float32)), \
                    torch.LongTensor(choose.astype(np.int32)), \
                    torch.from_numpy(img_masked.astype(np.float32)), \
                    torch.from_numpy(target.astype(np.float32)), \
                    torch.from_numpy(model_points.astype(np.float32)), \
                    torch.LongTensor([int(obj[idx]) - 1])
            else:
                target = np.zeros_like(model_points)
                return torch.from_numpy(cloud.astype(np.float32)), \
                    torch.LongTensor(choose.astype(np.int32)), \
                    torch.from_numpy(img_masked.astype(np.float32)), \
                    torch.from_numpy(target.astype(np.float32)), \
                    torch.from_numpy(model_points.astype(np.float32)), \
                    torch.LongTensor([int(obj[idx]) - 1])
        except Exception as err:
            return None, None, None, None, None, None

    def get_bounding_box(self, label):
        non_empty_rows = np.any(label, axis=1)
        non_empty_cols = np.any(label, axis=0)
        
        row_min, row_max = np.where(non_empty_rows)[0][[0, -1]]
        col_min, col_max = np.where(non_empty_cols)[0][[0, -1]]
        
        row_max += 1
        col_max += 1
        
        row_border = self.find_nearest_border(row_max - row_min)
        col_border = self.find_nearest_border(col_max - col_min)
        
        center_row = (row_min + row_max) // 2
        center_col = (col_min + col_max) // 2
        
        row_min, row_max = self.adjust_bbox(center_row, row_border)
        col_min, col_max = self.adjust_bbox(center_col, col_border)
        
        row_min, row_max = self.clamp_bbox(row_min, row_max, self.img_width)
        col_min, col_max = self.clamp_bbox(col_min, col_max, self.img_length)
        
        return row_min, row_max, col_min, col_max

    def find_nearest_border(self, size):
        for i, border in enumerate(self.border_list):
            if size > border and size < self.border_list[i + 1]:
                return self.border_list[i + 1]
        return size

    def adjust_bbox(self, center, border):
        min_border = center - border // 2
        max_border = center + border // 2
        return min_border, max_border

    def clamp_bbox(self, min_border, max_border, max_dimension):
        if min_border < 0:
            max_border += -min_border
            min_border = 0
        if max_border > max_dimension:
            min_border -= (max_border - max_dimension)
            max_border = max_dimension
        return min_border, max_border

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh_small
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)