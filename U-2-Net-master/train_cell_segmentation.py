# ============================================================================ #
# Author:GauthierLi                                                            #
# Date: 2022/07/09                                                             #
# Email: lwklxh@163.com                                                        #
# Description: train for cell segmentation.                                    #
# ============================================================================ #                     

import os
import pdb
from turtle import forward
import PIL
from isort import file
from matplotlib.pyplot import cla
from sqlalchemy import collate

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from model import U2NET, U2NETP
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>> 1 build_dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ============================================================================
class cell_seg_dataset(Dataset):
    def __init__(self, root:str, transform=None, file_list=None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        if file_list is  None:
            self.imgs = self._get_imgs_name(list(os.listdir(root)))[:3]
        else:
            self.imgs = file_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]
        img = os.path.join(self.root,image + ".png")
        img = Image.open(img)
        label = os.path.join(self.root, image + "_label.png")
        label = Image.open(label).convert("L")
        if self.transform is not None:
            img, label = self.transform(img), self.transform(label)
        return img, label

    @staticmethod
    def _get_imgs_name(raw_list:list) -> list:
        clean_list = []
        for img in raw_list:
            img_name = img.split(".")[0]
            if img_name.split("_")[-1] != "label":
                clean_list.append(img_name)
        return clean_list

    @staticmethod
    def collate_fn(batch):
        pass

def build_dataLoader(root, batch_size=32, split=0.8, transform=None):
    file_list = cell_seg_dataset._get_imgs_name(list(os.listdir(root)))
    random.shuffle(file_list)
    lth = len(file_list)
    train_cnt = int(lth * split)

    train_dataset = cell_seg_dataset(root, transform=transform, file_list=file_list[:train_cnt]) 
    val_dataset = cell_seg_dataset(root, transform=transform, file_list=file_list[train_cnt:]) 

    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=)
    val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_dataLoader, val_dataLoader

# ============================================================================================================= 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2,build_transforms <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# ============================================================================================================= 
def build_transforms():
    t = {"train":T.Compose(T.ToTensor(), T.RandomRotation(8), T.Resize((512, 512)), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
        "valid_test": T.Compose(T.ToTensor(), T.Resize((512, 512)), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))}
    return t

# 3,Augumentations ...
# 4,build_loss (focal_tversky_loss, bce_loss, dice_loss)
class Tversky(nn.Module):
    smooth = 1
    def __init__(self,alpha=0.3, gamma=0.7):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,pred:torch.Tensor, ground_truth:torch.Tensor):
        pred = pred.view(-1)
        ground_truth = ground_truth.view(-1)

        PG = pred @ ground_truth
        P_d_G = pred @ (1-ground_truth).T
        G_d_P = (1 - pred) @ ground_truth.T

        tversky = (PG + self.smooth) / (PG + self.alpha * P_d_G + self.gamma * G_d_P + self.smooth)
        return tversky

def tversky_loss(pred, ground_truth, alpha=0.3, gamma=0.7):
    return 1 - Tversky(alpha=alpha, gamma=gamma)(pred, ground_truth)

def bce_loss(pred, ground_truth):
    return nn.BCELoss()(pred, ground_truth)

def dice_loss(pred, ground_truth):
    return tversky_loss(pred, ground_truth,alpha=0.5, gamma=0.5)

def build_loss(pred, ground_truth, alpha=0.3, gamma=0.7):
    bce_loss = bce_loss(pred, ground_truth)
    dice_loss = dice_loss(pred, ground_truth)
    tversky_loss = tversky_loss(pred, ground_truth,alpha=alpha, gamma=gamma)
    return {"total_loss": 0.5 * (bce_loss + dice_loss) + 0.5 * tversky_loss,
    "bce_loss": bce_loss, "dice_loss":dice_loss, "tversky_loss":tversky_loss}


# 5,build_metrics (dice)

# 6,build_model (U2NET)
# 7,train_one_epoch
# 8,valid_one_epoch
# 9,test_one_epoch
# 10, CFG (lr, warm up with cosine annealing)

if __name__ == "__main__":
    train_dataLoader, val_dataLoader = build_dataLoader(r"F:\data\cell_segmentation\train_data")
    for img, label in train_dataLoader:
        loss = build_loss(img, label)
        print(loss["total_loss"].shape)
        break
