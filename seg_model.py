import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms, utils
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

import wandb

from dataset import SyntheticDataset, COLOR_PALETTE, collate_fn, NUM_OBJECTS, segmented2rgb

os.environ["WANDB_API_KEY"] = "d21415063a4a04fd70bb4f7728a855e8d4c29ba2"
wandb.login()

torch.cuda.manual_seed(123)


class Args:
    batch_size = 8
    cuda = True
    threads = 4
    num_epochs = 1000
    learning_rate = 0.0001
    model_dir = "/root/data/hand_object_segmentation/deeplab/training_results"
    checkpoints_dir = "/root/data/hand_object_segmentation/deeplab/checkpoints"

def log_image_table(images, ground_truth, predicted):
    table = wandb.Table(columns=["image", "predicted", "ground_truth"])

    for img, gt, pred in zip(images.detach().cpu().numpy(), ground_truth.detach().cpu().numpy(), predicted.detach().cpu().numpy()):
        img = np.transpose(img, (1, 2, 0))*255
        gt = segmented2rgb(gt)*255
        pred = segmented2rgb(pred)*255
        table.add_data(wandb.Image(img), wandb.Image(pred), wandb.Image(gt))
    wandb.log({"predictions_table":table}, commit=False)

def multi_acc(pred, label):
    _, tags = torch.max(pred, dim = 1)
    _, label = torch.max(label, dim = 1)
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

def custom_DeepLabv3(out_channel):
  # model = deeplabv3_resnet101(pretrained=True, progress=True)
  model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
  model.classifier = DeepLabHead(2048, out_channel)
  return model

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == "__main__":
    opt = Args()
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    wandb.init(
      project="6D-Segmentation", 
      name=f"DeepLabv3 - 2", 
      config={
      "architecture": "DeepLabv3",
      "dataset": "Synthetic"
      })

    training_data_dir = "/root/data/rohith/gnn/HW2/training_data/v2.2"
    split_dir = "/root/data/rohith/gnn/HW2/training_data/splits/v2"

    train_dataset = SyntheticDataset(training_data_dir, split_dir, mode="train")
    val_dataset = SyntheticDataset(training_data_dir, split_dir, mode="val")
    
    train_loader = DataLoader(dataset=train_dataset, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True, num_workers = 8)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=collate_fn, batch_size=opt.batch_size, shuffle=True, num_workers = 8)

    model = custom_DeepLabv3(out_channel = NUM_OBJECTS+3).to(device) 
    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr= opt.learning_rate)
    loss_fn = nn.CrossEntropyLoss().to(device) 

    epoch_losses_train = []
    epoch_losses_val = []
    best_acc = -1000
    for epoch in range(opt.num_epochs):
        model.train()
        batch_losses = []
        batch_acc = []
        for step, (imgs, label_imgs) in enumerate(train_loader):
            imgs = imgs.to(device).float() 
            label_imgs = label_imgs.to(device) 
            outputs = model(imgs) 
            outputs = outputs["out"]
            loss = loss_fn(outputs, label_imgs)
            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)
            acc = multi_acc(outputs, label_imgs).cpu().numpy()
            batch_acc.append(acc)
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            metrics = {"train/train_loss_step": loss, 
                       "train/train_acc_step": acc,
                       "train/step": step}
            wandb.log(metrics)

        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        epoch_losses_train.append(epoch_loss)
        
        train_metrics = {"train/train_loss": epoch_loss, 
                          "train/train_acc" : epoch_acc,
                          "train/epoch": epoch}
        wandb.log(train_metrics)
        print (f"Epoch: {epoch+1} Train Loss: {epoch_loss}, Train Acc: {epoch_acc}")

        model.eval()
        batch_losses = []
        batch_acc = []
        for step, (imgs, label_imgs) in enumerate(val_loader):
            with torch.no_grad():
                imgs = imgs.to(device).float() 
                label_imgs = label_imgs.to(device) 
                outputs = model(imgs)
                outputs = outputs["out"]
                loss = loss_fn(outputs, label_imgs)
                loss_value = loss.data.cpu().numpy()
                acc = multi_acc(outputs, label_imgs).cpu().numpy()
                batch_losses.append(loss_value)
                batch_acc.append(acc)
                metrics = {"val/val_loss_step": loss, 
                       "val/val_acc_step": acc,
                       "val/step": step}
                wandb.log(metrics)
                
        log_image_table(imgs, label_imgs, outputs)
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_acc)
        val_metrics = {"val/val_loss": epoch_loss, 
                       "val/val_accuracy": epoch_acc,
                       "val/epoch": epoch}
        wandb.log(val_metrics)
        epoch_losses_val.append(epoch_loss)
        print (f"Epoch: {epoch+1} Val Loss: {epoch_loss}, Val Acc: {epoch_acc}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            checkpoint_path = "/root/data/rohith/gnn/HW3/checkpoints/" +"deeplabv3_best" + ".pth"
            torch.save(model.state_dict(), checkpoint_path)