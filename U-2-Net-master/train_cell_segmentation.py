# ============================================================================ #
# Author:GauthierLi                                                            #
# Date: 2022/07/09                                                             #
# Email: lwklxh@163.com                                                        #
# Description: train for cell segmentation.                                    #
# ============================================================================ #                     

import os
import pdb
import PIL
import time
import torch
import random

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from zmq import device
from model import U2NET, U2NETP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

# ============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>> 1 build_dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ============================================================================
class cell_seg_dataset(Dataset):
    def __init__(self, root:str, transform=None, file_list=None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        if file_list is  None:
            self.imgs = self._get_imgs_name(list(os.listdir(root)))[:5]
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
            img, label = self.transform["img"](img), self.transform["label"](label)
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

def build_dataLoader(CFG, root, batch_size=32, split=0.8, transform=None):
    file_list = cell_seg_dataset._get_imgs_name(list(os.listdir(root)))
    random.shuffle(file_list)
    lth = len(file_list)
    train_cnt = int(lth * split)

    train_dataset = cell_seg_dataset(root, transform=transform, file_list=file_list[:train_cnt]) 
    val_dataset = cell_seg_dataset(root, transform=transform, file_list=file_list[train_cnt:]) 

    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=default_collate, num_workers=CFG.num_workers)
    val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=default_collate, num_workers=CFG.num_workers)

    return train_dataLoader, val_dataLoader

# ============================================================================================================= 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2,build_transforms <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# ============================================================================================================= 
def build_transforms(CFG):
    rand_rot = T.RandomRotation(8)
    t = {"train":{"img":T.Compose([T.ToTensor(),rand_rot,  T.Resize(CFG.img_size)]),
        "label":T.Compose([T.ToTensor(),rand_rot,  T.Resize(CFG.img_size)])},
        "valid_test":{"img": T.Compose([T.ToTensor(), T.Resize(CFG.img_size)]),
        "label": T.Compose([T.ToTensor(), T.Resize(CFG.img_size)])}
        }
    return t


# 3,Augumentations ...
# 4,build_loss (focal_tversky_loss, bce_loss, dice_loss)
class Tversky(nn.Module):
    smooth = 1
    def __init__(self,alpha=0.3, gamma=0.7):
        super(Tversky, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self,pred:torch.Tensor, ground_truth:torch.Tensor):
        pred = pred.view(-1)
        ground_truth = ground_truth.view(-1)

        PG = torch.sum(pred * ground_truth)
        P_d_G = torch.sum(pred * (1-ground_truth))
        G_d_P = torch.sum((1 - pred) * ground_truth)

        tversky = (PG + self.smooth) / (PG + self.alpha * P_d_G + self.gamma * G_d_P + self.smooth)
        return tversky

def build_tversky(alpha=0.3, gamma=0.7):
    return Tversky(alpha=alpha, gamma=gamma)

def build_bce_loss():
    return nn.BCEWithLogitsLoss()

def build_dice_loss():
    return build_tversky(alpha=0.5, gamma=0.5)

def build_loss( alpha=0.3, gamma=0.7):
    bce_loss = build_bce_loss()
    dice_loss = build_dice_loss()
    tversky_loss = build_tversky(alpha=alpha, gamma=gamma)
    return {"bce_loss": bce_loss, "dice_loss":dice_loss, "tversky_loss":tversky_loss}


# =============================================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5,build_metrics (dice) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# build from 1 - dice_loss
# =============================================================================================================


# =============================================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6,build_model (U2NET) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================================================
def build_model():
    return U2NET(in_ch=3, out_ch=1)

# =============================================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7,train_one_epoch <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================================================
def train_one_epoch(CFG, model, train_loader, optimizer):
    model.train()
    model.to(CFG.device)
    total_loss, total_bce_loss, total_tversky_loss = 0., 0., 0.
    scaler = torch.cuda.amp.GradScaler()

    avg_dice_score = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")

    for _, (images, masks) in pbar:
        images = images.to(CFG.device, dtype=torch.float)
        masks = masks.to(CFG.device, dtype=torch.float)
        losses = build_loss()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images) # 7 feature maps tuple
            tmp_dice_list = []
            for i in range(7):
                bce_LV = losses["bce_loss"](outputs[i], masks) # Loss Value
                tversky_LV = 1 - losses["tversky_loss"](outputs[i], masks)
                dice_LV = 1 - losses["dice_loss"](outputs[i], masks)

                total_LV = 0.5 * (bce_LV + dice_LV) + 0.5 * tversky_LV
                # pdb.set_trace()

                tmp_dice_list.append((1 - dice_LV).cpu().detach().numpy())
        
            avg_dice_score.append(np.mean(np.array(tmp_dice_list)))
        
        # total_LV.backward()
        # optimizer.step()
        scaler.scale(total_LV).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += total_LV.item() / (7 * images.shape[0])
        total_bce_loss += total_LV.item() / (7 * images.shape[0])
        total_tversky_loss += tversky_LV.item() / (7 * images.shape[0])

    train_dice_score = np.mean(avg_dice_score)
    print("Training dice: {:.4f}".format(train_dice_score), flush=True)

    current_lr = optimizer.param_groups[0]['lr']
    print("lr:{:.6f}".format(current_lr), flush=True)

    print("loss : {:.3f}, bce : {:.3f}, tversky : {:.3f}".format(total_loss, total_bce_loss, total_tversky_loss), flush=True)



# 8,valid_one_epoch
# 9,test_one_epoch
# 10, CFG (lr, warm up with cosine annealing)

# ===========================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================================


if __name__ == "__main__":
    class CFG:
        epoch = 150
        lr = 1e-3
        wd = 1e-5
        lr_drop = 10

        train_bs = 4
        num_workers=8
        valid_bs = train_bs * 2
        img_size = (320,320)

        # inference
        thr = 0.5
        early_stop = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_data = r"F:\data\cell_segmentation\train_data"
        test_data = r"F:\data\cell_segmentation\test_data"
        ckpt_path = os.path.join(os.getcwd(), f"ckpt_U2Net_{epoch}_{img_size[0] * img_size[1]}_thr{thr}")

    # model = build_model().to(CFG.device)
    # train_transforms = build_transforms(CFG)["train"]
    # train_dataLoader, val_dataLoader = build_dataLoader(r"F:\data\cell_segmentation\train_data", batch_size=CFG.train_bs, transform=train_transforms)

    # for img, label in train_dataLoader:
    #     img = img.to(CFG.device)
    #     label = label.to(CFG.device)
    #     pred = model(img)
    #     pdb.set_trace()
    #     break

    
    train_flag = True
    if train_flag:
        model = build_model()
        loss_dict = build_loss()
        train_transforms = build_transforms(CFG)["train"]
        train_dataLoader, val_dataLoader = build_dataLoader(CFG,r"F:\data\cell_segmentation\train_data", batch_size=CFG.train_bs, transform=train_transforms)
        
        best_val_dice = 0
        best_epoch = 0

        for epoch in range(CFG.epoch):
            optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:1/(epoch+1))
            start_time = time.time()

            train_one_epoch(CFG, model=model, train_loader=train_dataLoader,optimizer=optimizer)


    # for img, label in train_dataLoader:
    #     B,C,H,W = img.shape
    #     fake_pred = torch.ones((B,1,H, W))
    #     loss = build_loss(fake_pred, label)
    #     print(loss["total_loss"])
    #     break
