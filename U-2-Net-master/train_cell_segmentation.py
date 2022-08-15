# ============================================================================ #
# Author:GauthierLi                                                            #
# Date: 2022/07/09                                                             #
# Email: lwklxh@163.com                                                        #
# Description: train for cell segmentation.                                    #
# ============================================================================ #                     

import os
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
from sklearn.model_selection import StratifiedGroupKFold
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
            self.imgs = file_list

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]
        img = os.path.join(self.root,image + ".png")
        img = Image.open(img).convert("RGB")
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
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=default_collate, num_workers=CFG.num_workers)

    if split != 1:
        val_dataset = cell_seg_dataset(root, transform=transform, file_list=file_list[train_cnt:]) 
        val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=default_collate, num_workers=CFG.num_workers)

        return train_dataLoader, val_dataLoader
    else:
        return train_dataLoader

# ============================================================================================================= 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2,build_transforms <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
# ============================================================================================================= 
def build_transforms(CFG):
    rand_rot = T.RandomRotation(3)
    t = {"train":{"img":T.Compose([T.ToTensor(),rand_rot,  T.Resize(CFG.img_size)]),
        "label":T.Compose([T.ToTensor(),rand_rot,  T.Resize(CFG.img_size)])},
        "valid_test":{"img": T.Compose([T.ToTensor(), T.Resize(CFG.img_size)]),
        "label": T.Compose([T.ToTensor(), T.Resize(CFG.img_size)])}
        }
    return t


# 3,Augumentations ...
# 4,build_loss (focal_tversky_loss, bce_loss, dice_loss)
class Tversky(nn.Module):
    smooth = 1e-5
    def __init__(self,alpha=0.3, gamma=0.7, reduce=True):
        super(Tversky, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self,pred:torch.Tensor, ground_truth:torch.Tensor):
        batch, _, _, _ = pred.shape
        pred_view = pred.view(batch, -1)
        ground_truth_view = ground_truth.view(batch, -1)
        
        PG = (pred_view * ground_truth_view).sum(axis=1)
        P_d_G = (pred_view * (1 - ground_truth_view)).sum(axis=1)
        G_d_P = ((1 - pred_view) * ground_truth_view).sum(axis=1)

        tversky = (PG + self.smooth) / (PG + self.alpha * P_d_G + self.gamma * G_d_P + self.smooth)
        tversky_mean = tversky.mean()
    
        if self.reduce:
            return tversky_mean
        else:
            return tversky

class Tversky_loss(nn.Module):
    def __init__(self,alpha=0.3, gamma=0.7):
        super(Tversky_loss, self).__init__()
        self.tversky = Tversky(alpha=alpha, gamma=gamma)

    def forward(self,pred:torch.Tensor, ground_truth:torch.Tensor):
        return torch.pow(1 - self.tversky(pred, ground_truth), 0.75)

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()
        self.tversky = Tversky(alpha=0.5, gamma=0.5)

    def forward(self,pred:torch.Tensor, ground_truth:torch.Tensor):
        return 1 - self.tversky(pred, ground_truth)

def build_bce_loss():
    return torch.nn.BCELoss()

def build_dice(reduce=True):
    return Tversky(alpha=0.5, gamma=0.5, reduce=reduce)

def build_loss( alpha=0.3, gamma=0.7):
    bce_loss = build_bce_loss()
    dice_loss = Dice_loss()
    tversky_loss = Tversky_loss(alpha=alpha, gamma=gamma)
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
        viewer(images, masks, outputs[0])
        tmp_dice_list = []


        total_LV = 0
        for i in range(len(outputs)):
            bce_LV = losses["bce_loss"](outputs[i], masks) # Loss Value
            tversky_LV = losses["tversky_loss"](outputs[i], masks)
            dice_LV = losses["dice_loss"](outputs[i], masks)

            bags_loss = bce_LV + tversky_LV + dice_LV
            total_LV += bags_loss

            tmp_dice_list.append(1 - dice_LV.item())

        total_LV.backward()
        optimizer.step()
        # scaler.scale(total_LV).backward()
        # scaler.step(optimizer)
        # scaler.update()
        

        avg_dice_score.append(np.mean(tmp_dice_list))
        # tqdm.write("loss : {:.3f}, bce : {:.3f}, tversky : {:.3f}, dice : {:.3f}".format(total_loss, total_bce_loss, total_tversky_loss, avg_dice_score))
    

        print("\nloss : {:.3f}, dice_score : {:.3f}".format(total_LV.item(), tmp_dice_list[-1]), flush=True)


    train_dice_score = np.mean(avg_dice_score)

    current_lr = optimizer.param_groups[0]['lr']
    print("lr:{:.6f}".format(current_lr), flush=True)
    print("Training dice: {:.4f}".format(train_dice_score), flush=True)



# 8,valid_one_epoch
@torch.no_grad()
def valid_one_epoch(CFG,  model, valid_loader):
    model.to(CFG.device)
    model.eval()
    dice = []
    metric = build_dice(reduce = False)
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Valid", file=sys.stdout)
    for _, (images, masks) in pbar:
        images = images.to(CFG.device)
        masks = masks.to(CFG.device)

        outputs = model(images)

        for i in range(len(outputs)):
            dice += metric(outputs[i], masks).cpu().detach().numpy().tolist()
    
    dice = np.mean(np.array(dice))
    print("Valid dice {:.4f}".format(dice), flush=True)
    return dice
    

# 9,test_one_epoch
@torch.no_grad()
def test_one_epoch(CFG, model, test_loader, view=True):
    model.to(CFG.device)
    ckpt = os.path.join(CFG.ckpt_path, "best_epoch_235.pth")
    assert os.path.isfile(ckpt), "checkpoints not exists ... ..."
    state_dict = torch.load(ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    dice = []
    metric = build_dice(reduce=False)
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Test", file=sys.stdout)
    for _, (images, masks) in pbar:
        images = images.to(CFG.device)
        masks = masks.to(CFG.device)

        outputs = model(images)
        if CFG.thr:
            outputs[0][outputs[0] > CFG.thr] = 1
            outputs[0][outputs[0] < CFG.thr] = 0
        # pdb.set_trace()
        viewer(images, masks, outputs[0], waitKey=1)

        dice += metric(outputs[0], masks).cpu().detach().numpy().tolist()
    
    dice = np.mean(np.array(dice))
    print("Test dice {:.4f}".format(dice), flush=True)
    return dice



# ===========================================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> utils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===========================================================================================
def  viewer(img, mask, pred, waitKey=1):
    img = (img.cpu().detach().numpy()[0].transpose((1,2,0)) * 255).astype("uint8")
    mask = (mask[0].squeeze().cpu().detach().numpy() * 255).astype("uint8")
    pred = (pred[0].squeeze().cpu().detach().numpy() * 255).astype("uint8")
    zero_ = np.zeros_like(mask)
    
    mask = np.stack([mask, zero_, mask], axis=2)
    pred = np.stack([zero_, pred, zero_], axis=2)

    alpha = 0.255
    mask = (alpha * mask + (1 - alpha) * img).astype("uint8")
    pred = (alpha * pred + (1 - alpha) * img).astype("uint8")


    total = np.hstack([img, mask, pred])
    cv2.imshow("view", total)
    cv2.waitKey(waitKey)



if __name__ == "__main__":
    class CFG:
        seed = 9422
        epoch = 150
        lr = 5e-4
        wd = 1e-6
        lr_drop = 10

        train_bs = 4
        num_workers=0
        valid_bs = train_bs * 2
        img_size = (512,512)
        # img_size = (320, 320)

        n_fold = 4

        # inference
        thr = 0.5
        early_stop = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_data = r"F:\data\cell_segmentation\train_data"
        test_data = r"F:\data\cell_segmentation\test_data"
        # ckpt_path = os.path.join(os.getcwd(), f"ckpt_U2Net_{epoch}_{img_size[0] * img_size[1]}_thr{thr}_{n_fold}fold")
        ckpt_path = r"F:\code\cell_seg\ckpt_U2Net_150_102400_thr0.5(version1 try 1)"

        resume = False
        warm_up_epoch = 20
        resume_path = r"F:\code\cell_seg\ckpt_U2Net_150_102400_thr0.5(current best)\epoch_70.pth"

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
    
    model = build_model()
    train_flag = False
    if train_flag:
        if CFG.resume:
            assert os.path.isfile(CFG.resume_path), "resume not exist ... ..."
            state_dict = torch.load(CFG.resume_path)
            model.load_state_dict(state_dict)
        loss_dict = build_loss()
        train_transforms = build_transforms(CFG)["train"]

        skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)


        train_dataLoader, val_dataLoader = build_dataLoader(CFG,CFG.train_data, batch_size=CFG.train_bs, transform=train_transforms)
        
        lr_scheduler_flag = True
        
        best_val_dice = 0
        best_epoch = 0

        best_epoch = 0
        for epoch in range(CFG.epoch):
            if epoch < CFG.warm_up_epoch:
                epoch_percent = epoch / float(CFG.epoch)
                optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr * epoch_percent, weight_decay=CFG.wd)
            else:
                if lr_scheduler_flag:
                    lr_scheduler_flag = False
                    optimizer = torch.optim.AdamW(model.parameters(), lr = CFG.lr , weight_decay=CFG.wd)
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=2e-6)
            start_time = time.time()

            train_one_epoch(CFG, model=model, train_loader=train_dataLoader,optimizer=optimizer)

            if not lr_scheduler_flag:
                lr_scheduler.step()
            dice = valid_one_epoch(CFG, model=model, valid_loader=val_dataLoader)

            if dice > best_val_dice:
                print("Saving best epoch ... ....", flush=True)
                best_val_dice = dice
                torch.save(model.state_dict(),os.path.join(CFG.ckpt_path, "best_epoch.pth"))
            
            print("saving last epoch ... ...", flush=True)
            torch.save(model.state_dict(),os.path.join(CFG.ckpt_path, f"epoch_{epoch}.pth"))
            if os.path.isfile(os.path.join(CFG.ckpt_path, f"epoch_{epoch - 1}.pth")):
                os.remove(os.path.join(CFG.ckpt_path, f"epoch_{epoch - 1}.pth"))

            epoch_time = time.time() - start_time
            print("epoch:{}, time:{:.2f}s\n".format(epoch, epoch_time), flush=True)
    
    test_flag = True
    if test_flag:
        model = build_model()
        transform = build_transforms(CFG)["valid_test"]
        test_dataloader = build_dataLoader(CFG, CFG.test_data,batch_size=1, transform=transform, split=1)
        train_dataLoader, val_dataLoader = build_dataLoader(CFG,CFG.train_data, batch_size=CFG.train_bs, transform=transform)
        test_one_epoch(CFG, model, test_dataloader)