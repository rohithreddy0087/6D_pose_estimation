import numpy as np
import os
import wandb
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from dataset import PoseDataset, collate_fn
from model import PoseNet
from loss import Loss

torch.cuda.manual_seed(123)
os.environ["WANDB_API_KEY"] = "d21415063a4a04fd70bb4f7728a855e8d4c29ba2"
wandb.login()

class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='simple distributed training job')
        self.parser.add_argument('--name', type=str, default='Expt - 3')
        self.parser.add_argument('--data_dir', type=str, default='/root/data/gnn_adjacency/data/n5_dataset')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Input batch size (default: 16)')
        self.parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
        self.parser.add_argument('--threads', type=int, default=4, help='Number of threads for data loading (default: 4)')
        self.parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs to train (default: 1000)')
        self.parser.add_argument('--save_every', type=int, default=50, help='Save checkpoints for every 10 epochs')
        self.parser.add_argument('--lr', default=0.0001, help='learning rate')
        self.parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
        self.parser.add_argument('--w', default=0.015, help='learning rate')
        self.parser.add_argument('--w_rate', default=0.4, help='learning rate decay rate')
        self.parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
        self.parser.add_argument('--checkpoints_dir', type=str, default='/root/data/rohith/gnn/HW2/dense_fusion/check_points', help='Checkpoints directory')

    def parse_args(self):
        return self.parser.parse_args()

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        wandb,
        opt,
        save_every: int,
    ) -> None:
        self.model = model.cuda()
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.criterion = Loss(opt.num_points_mesh, opt.sym_list)
        self.wandb = wandb
        self.opt = opt

    def _run_epoch(self, epoch, mode = "train"):
        if mode == "train":
            data = self.train_data
            self.model.train()
        else:
            data = self.val_data
            self.model.eval()
            best_test = np.Inf
        
        train_count = 0
        train_dis_avg = 0.0
        batch_losses = []
        for step, (cloud, choose, img, target, model_points, idx) in enumerate(data):
            points, choose, img, target, model_points, idx = Variable(cloud).cuda(), \
                                                                Variable(choose).cuda(), \
                                                                Variable(img).cuda(), \
                                                                Variable(target).cuda(), \
                                                                Variable(model_points).cuda(), \
                                                                Variable(idx).cuda()

            pred_r, pred_t, pred_c, emb = self.model(img, points, choose, idx)
            loss, dis, new_points, new_target = self.criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, self.opt.w, False)
            
            train_dis_avg += dis.item()
            train_count += 1

            if mode == "train":
                loss.backward()
                if train_count % self.opt.batch_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    batch_avg = train_dis_avg/self.opt.batch_size
                    print(f"Step {step} | {mode} Loss: {batch_avg}")
                    batch_losses.append(batch_avg)
                    batch_metrics = {f"{mode}/batch_loss": batch_avg,
                                    f"{mode}/batch_step": step}
                    if batch_avg < 1:
                        self.wandb.log(batch_metrics)
                    train_dis_avg = 0
            else:
                batch_losses.append(dis.item())
        
        if epoch in [ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
            print(optimizer)
        
        epoch_loss = np.mean(batch_losses)
        if mode == "val" and epoch_loss <= best_test:
            best_test = epoch_loss
            self._save_checkpoint(epoch, "best")
        metrics = {f"{mode}/loss": epoch_loss,
                    f"{mode}/step": epoch}
        print(f"Epoch {epoch} | {mode} Loss: {epoch_loss}")
        self.wandb.log(metrics)
        
    def _save_checkpoint(self, epoch, tag = "None"):
        ckp = self.model.state_dict()
        if tag == "best":
            checkpoint_path = self.opt.checkpoints_dir + "/expt-5" +"_best" + ".pth"
        else:
            checkpoint_path = self.opt.checkpoints_dir + "/expt-5" +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(ckp, checkpoint_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch, "train")
            self._run_epoch(epoch, "val")
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs(opt):
    training_data_dir = "/root/data/rohith/gnn/HW2/training_data/v2.2"
    split_dir = "/root/data/rohith/gnn/HW2/training_data/splits/v2"
    objects_csv_train = "/root/data/rohith/gnn/HW2/training_data/objects_v1.csv"

    train_dataset = PoseDataset(training_data_dir, split_dir, objects_csv_train, mode="train")
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers = 16)
    print(f"Length of training dataset {len(train_dataset)}")
    opt.sym_list = train_dataset.get_sym_list()
    opt.num_points_mesh = train_dataset.get_num_points_mesh()

    val_dataset = PoseDataset(training_data_dir, split_dir, objects_csv_train, mode="val")
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1, shuffle=False, num_workers = 16)
    print(f"Length of training dataset {len(val_dataset)}")
    opt.num_points = 1000
    opt.num_objects = 79
    model = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr )
    return train_dataloader, val_dataloader, model, optimizer


def main(opt, wandb):
    train_dataloader, val_dataloader, model, optimizer = load_train_objs(opt)
    trainer = Trainer(model, train_dataloader, val_dataloader, optimizer, wandb, opt, opt.save_every)
    trainer.train(opt.num_epochs)

if __name__ == "__main__":
    opt = Args().parse_args()
    print(opt)
    run_wandb = wandb.init(
      project="6D Pose Estimation", 
      name=opt.name, 
      config={
      "architecture": "Dense Fusion",
      "dataset": "Synthetic dataset",
      })
    main(opt, run_wandb)