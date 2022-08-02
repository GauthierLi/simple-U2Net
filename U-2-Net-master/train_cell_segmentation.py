# ============================================================================ #
# Author:GauthierLi                                                            #
# Date: 2022/07/09                                                             #
# Email: lwklxh@163.com                                                        #
# Description: train for cell segmentation.                                    #
# ============================================================================ #                     

from genericpath import isfile
import os
from subprocess import check_output
import cv2
import pdb
import sys
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
        if file_list is None:
            self.imgs = self._get_imgs_name(list(os.listdir(root)))
        else:
            self.imgs = file_list[:3]

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
        P_d_G = torch.sum(pred * (1 - ground_truth))
        G_d_P = torch.sum((1 - pred) * ground_truth)

        tversky = (PG + self.smooth) / (PG + self.alpha * P_d_G + self.gamma * G_d_P + self.smooth)
        return tversky

def build_tversky(alpha=0.3, gamma=0.7):
    return Tversky(alpha=alpha, gamma=gamma)

def build_bce_loss():
    return nn.BCEWithLogitsLoss()

def build_dice():
    return build_tversky(alpha=0.5, gamma=0.5)

def build_loss( alpha=0.3, gamma=0.7):
    bce_loss = build_bce_loss()
    dice = build_dice()
    tversky = build_tversky(alpha=alpha, gamma=gamma)
    return {"bce_loss": bce_loss, "dice":dice, "tversky":tversky}


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
    # scaler = torch.cuda.amp.GradScaler()

    avg_dice_score = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train", file=sys.stdout)

    for _, (images, masks) in pbar:
        images = images.to(CFG.device, dtype=torch.float)
        masks = masks.to(CFG.device, dtype=torch.float)
        losses = build_loss()

        optimizer.zero_grad()

        # with torch.cuda.amp.autocast(enabled=True):
        outputs = model(images) # 7 feature maps tuple
        viewer(images,masks, outputs[6])
        tmp_dice_list = []

        for i in range(len(outputs)):
            bce_LV = losses["bce_loss"](outputs[i], masks) # Loss Value
            tversky_LV = torch.pow(1 - losses["tversky"](outputs[i], masks), 0.75)
            dice_LV = 1 - losses["dice"](outputs[i], masks)

            total_LV = bce_LV + dice_LV +  tversky_LV

            tmp_dice_list.append(1 - dice_LV.item())
            total_loss += total_LV.item()
            total_bce_loss += bce_LV.item()
            total_tversky_loss += tversky_LV.item()

        avg_dice_score.append(np.mean(np.array(tmp_dice_list)))
        # tqdm.write("loss : {:.3f}, bce : {:.3f}, tversky : {:.3f}, dice : {:.3f}".format(total_loss, total_bce_loss, total_tversky_loss, avg_dice_score))
    
        
        total_LV.backward()
        optimizer.step()
        # scaler.scale(total_LV).backward()
        # scaler.step(optimizer)
        # scaler.update()

        total_loss = total_loss / (7 * images.shape[0])
        total_bce_loss = total_bce_loss / (7 * images.shape[0])
        total_tversky_loss = total_tversky_loss / (7 * images.shape[0])


    train_dice_score = np.mean(avg_dice_score)

    current_lr = optimizer.param_groups[0]['lr']
    print("lr:{:.6f}".format(current_lr), flush=True)
    print("Training dice: {:.4f}".format(train_dice_score), flush=True)

    print("loss : {:.3f}, bce : {:.3f}, tversky : {:.3f}".format(total_loss, total_bce_loss, total_tversky_loss), flush=True)


# 8,valid_one_epoch
@torch.no_grad()
def valid_one_epoch(CFG,  model, valid_loader):
    model.to(CFG.device)
    model.eval()
    dice = []
    metric = build_dice()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid", file=sys.stdout)
    for _, (images, masks) in pbar:
        images = images.to(CFG.device)
        masks = masks.to(CFG.device)

        outputs = model(images)

        for i in range(len(outputs)):
            dice.append(metric(outputs[i], masks).item())
    
    dice = np.mean(np.array(dice))
    print("Valid dice {:.4f}".format(dice), flush=True)
    return dice
    

# 9,test_one_epoch

# ===========================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================================
def  viewer(img, mask, pred):
    img = (img.cpu().detach().numpy()[0].transpose((1,2,0)) * 255).astype("uint8")
    mask = (mask[0].squeeze().cpu().detach().numpy() * 255).astype("uint8")
    pred = (pred[0].squeeze().cpu().detach().numpy() * 255).astype("uint8")
    
    mask = np.stack([mask, mask, mask], axis=2)
    pred = np.stack([pred, pred, pred], axis=2)

    total = np.hstack([img, mask, pred])
    cv2.imshow("view", total)
    cv2.waitKey(1)



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
    if not os.path.exists(CFG.ckpt_path):
        os.makedirs(CFG.ckpt_path)
    
    
    train_flag = True
    if train_flag:
        model = build_model()
        loss_dict = build_loss()
        train_transforms = build_transforms(CFG)["train"]
        train_dataLoader, val_dataLoader = build_dataLoader(CFG,r"F:\data\cell_segmentation\train_data", batch_size=CFG.train_bs, transform=train_transforms)
        
        best_val_dice = 0
        best_epoch = 0

        best_epoch = 0
        for epoch in range(CFG.epoch):
            optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,lr_lambda=lambda epoch:1/(epoch+1))
            start_time = time.time()

            train_one_epoch(CFG, model=model, train_loader=train_dataLoader,optimizer=optimizer)
            dice = valid_one_epoch(CFG, model=model, valid_loader=val_dataLoader)

            if dice > best_val_dice:
                print("Saving best epoch ... ....", flush=True)
                torch.save(model.state_dict(),os.path.join(CFG.ckpt_path, "best_epoch.pth"))
            
            print("saving last epoch ... ...", flush=True)
            torch.save(model.state_dict(),os.path.join(CFG.ckpt_path, f"epoch_{epoch}.pth"))
            if os.path.isfile(os.path.join(CFG.ckpt_path, f"epoch_{epoch - 1}.pth")):
                os.remove(os.path.join(CFG.ckpt_path, f"epoch_{epoch - 1}.pth"))

            epoch_time = time.time() - start_time
            print("epoch:{}, time:{:.2f}s\n".format(epoch, epoch_time), flush=True)
                
