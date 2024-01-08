import numpy as np
import json
import torch
import argparse
from torch.autograd import Variable

from model import PoseNet
from loss import Loss
from test_dataset import PoseDataset, NUM_OBJECTS, collate_fn
from utils import quaternion_matrix

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='simple distributed training job')
        self.parser.add_argument('--name', type=str, default='Expt - 1')
        self.parser.add_argument('--data_dir', type=str, default='/root/data/gnn_adjacency/data/n5_dataset')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Input batch size (default: 16)')
        self.parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
        self.parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loading (default: 4)')
        self.parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to train (default: 1000)')
        self.parser.add_argument('--save_every', type=int, default=50, help='Save checkpoints for every 10 epochs')
        self.parser.add_argument('--lr', default=0.0001, help='learning rate')
        self.parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
        self.parser.add_argument('--w', default=0.015, help='learning rate')
        self.parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
        self.parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/root/data/rohith/gnn/HW2/dense_fusion/check_points', help='Checkpoints directory')

    def parse_args(self):
        return self.parser.parse_args()

opt = Args().parse_args()
training_data_dir = "/root/data/rohith/gnn/HW2/testing_data/v2.2"
split_dir = "/root/data/rohith/gnn/HW2/testing_data"
objects_csv_train = "/root/data/rohith/gnn/HW2/testing_data/objects_v1.csv"
out_file = "test_nn.json"

# training_data_dir = "/root/data/rohith/gnn/HW2/training_data/v2.2"
# split_dir = "/root/data/rohith/gnn/HW2/training_data/splits/v2"
# objects_csv_train = "/root/data/rohith/gnn/HW2/training_data/objects_v1.csv"
# out_file = "val_nn.json"

checkpoint_path = "/root/data/rohith/gnn/HW2/dense_fusion/check_points/expt-4_best.pth"

dataset = PoseDataset(training_data_dir, split_dir, objects_csv_train, mode="test")
print(f"Length of training dataset {len(dataset)}")
opt.num_points = 1000
opt.num_objects = 79
model = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
model = model.cuda()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

opt.sym_list = dataset.get_sym_list()
opt.num_points_mesh = dataset.get_num_points_mesh()

criterion = Loss(opt.num_points_mesh, opt.sym_list)
json_out = {}

for i in range(len(dataset)):
    meta = dataset.load_pickle(dataset.meta_files[i])
    prefix = dataset.prefixes[i]
    json_out[prefix] = {"poses_world":[None]*NUM_OBJECTS}
    print(i)
    for idx, (obj_name, obj_id) in enumerate(zip(meta['object_names'], meta['object_ids'])):
        cloud, choose, img, target, model_points, idx = dataset.getitem(i, idx)
        
        if cloud is  None:
            pose = np.eye(4)
        else:
            
            points, choose, img, target, model_points, idx = Variable(cloud[None]).cuda(), \
                                                                    Variable(choose[None]).cuda(), \
                                                                    Variable(img[None]).cuda(), \
                                                                    Variable(target[None]).cuda(), \
                                                                    Variable(model_points[None]).cuda(), \
                                                                    Variable(idx[None]).cuda()
            pred_r, pred_t, pred_c, emb = model(img, points, choose, idx)

            pred_r, pred_t, pred_c, emb = model(img, points, choose, idx)

            pred_r_norm = torch.norm(pred_r, dim=2).view(1, opt.num_points, 1)
            pred_r = pred_r / pred_r_norm

            pred_c = pred_c.view(1, opt.num_points)
            _, max_pos = torch.max(pred_c, 1)

            pred_t = pred_t.view(1 * opt.num_points, 1, 3)
            points = points.view(1 * opt.num_points, 1, 3)
            final_t = (points + pred_t)[max_pos[0]].view(-1).cpu().data.numpy()

            final_r = pred_r[0][max_pos[0]].view(-1).cpu().data.numpy()

            pose = quaternion_matrix(final_r)

            my_pred = np.concatenate((final_r, final_t))

        json_out[prefix]["poses_world"][obj_id] = pose.tolist()

    json_data = json.dumps(json_out)
    with open(out_file, 'w') as json_file:
        json_file.write(json_data)
