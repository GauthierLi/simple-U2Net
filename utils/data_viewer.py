from cProfile import label
import os
import pdb
from tkinter.messagebox import NO
import cv2

import numpy as np

class data_viewer:
    def __init__(self, data_path:str) -> None:
        self.path = data_path
        self.imgs = self._get_imgs_name(list(os.listdir(data_path)))

    def _get_imgs_name(self,raw_list:list) -> list:
        clean_list = []
        for img in raw_list:
            img_name = img.split(".")[0]
            if img_name.split("_")[-1] != "label":
                clean_list.append(img_name)
        return clean_list

    @staticmethod
    def view(img_path:str, label_path:str) -> None:
        """which label is a single channel binary mask image"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        H,W = label.shape
        pdb.set_trace()
        blank = np.zeros((H,W)).astype("uint8")
        label = np.stack([blank, blank, label],axis=2)

        # pdb.set_trace()

        img_with_label = cv2.addWeighted(img, 0.8, label, 0.2, 0)
        cv2.imshow("img", img_with_label)
        cv2.waitKey(0)



if __name__ == "__main__":
    class CFG:
        path = r"F:\data\cell_segmentation\train_data"
    
    # test
    t1 = data_viewer(CFG.path)
    for img in t1.imgs:
        t1.view(os.path.join(t1.path, img + ".png"), os.path.join(t1.path, img + "_label.png"))
    