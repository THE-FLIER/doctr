import collections
import os
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import hdf5storage as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from torch.utils.data import DataLoader
import json
import pdb
import albumentations as A
random.seed(12345678)

class Doc3dDataset(Dataset):
    def __init__(self, root, img_size=512, is_augment=False, split="val"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = os.path.expanduser(root)
        self.split = split
        self.is_augment = is_augment
        self.files = collections.defaultdict(list)
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.augmentation = A.Compose(
            [
                A.ColorJitter(),
            ]
        )

        if self.split == "train":
            path = os.path.join(self.root, "train/", split + "_1.txt")
            with open(path, 'r') as f:
                file_list = tuple(i.strip() for i in f.readlines())
                self.files[split] = file_list
        elif self.split == "val":
            path = os.path.join(self.root, "val/", split + "_1.txt")
            with open(path, 'r') as f:
                file_list = tuple(i.strip() for i in f.readlines())
                self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # name_of_img
        im_name = self.files[self.split][index]

        # img
        im_path = os.path.join(self.root, f"{self.split}/img/", f"{im_name}.png")
        image = cv2.imread(im_path)

        # backward map
        bm_path = os.path.join(self.root, f"{self.split}/bm/", f"{im_name}.mat")
        bm = h5.loadmat(bm_path)["bm"]

        # 3D Coordinates
        wc_path = os.path.join(self.root, f"{self.split}/wc/" + f"{im_name}.exr")
        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        image, bm = self.transform(wc, bm, image)
        return image, bm

    def tight_crop(self, wc: np.ndarray):
        mask = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0) & (wc[:, :, 2] != 0)).astype(
            np.uint8
        )
        mask_size = mask.shape
        [y, x] = mask.nonzero()
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)

        wc = wc[min_y : max_y + 1, min_x : max_x + 1, :]
        s = 10
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), "constant")

        cx1 = random.randint(0, 2 * s)
        cx2 = random.randint(0, 2 * s) + 1
        cy1 = random.randint(0, 2 * s)
        cy2 = random.randint(0, 2 * s) + 1

        wc = wc[cy1:-cy2, cx1:-cx2, :]

        top: int = min_y - s + cy1
        bottom: int = mask_size[0] - max_y - s + cy2
        left: int = min_x - s + cx1
        right: int = mask_size[1] - max_x - s + cx2

        top = np.clip(top, 0, mask_size[0])
        bottom = np.clip(bottom, 1, mask_size[0] - 1)
        left = np.clip(left, 0, mask_size[1])
        right = np.clip(right, 1, mask_size[1] - 1)

        return wc, top, bottom, left, right
    def transform(self, wc, bm, img):
        wc, top, bottom, left, right = self.tight_crop(wc)

        img = img[top:-bottom, left:-right, :]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_augment:
            img = self.augmentation(image=img)["image"]

        # resize image
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        # resize bm
        bm = bm.astype(np.float32)
        bm[:, :, 1] = bm[:, :, 1] - top
        bm[:, :, 0] = bm[:, :, 0] - left
        bm = bm / np.array([448.0 - left - right, 448.0 - top - bottom])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.img_size[0], self.img_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size[0], self.img_size[1]))

        bm = np.stack([bm0, bm1], axis=-1)

        img = torch.from_numpy(img).float()
        bm = torch.from_numpy(bm).float()

        return img, bm
if __name__ == "__main__":
    data_path = "dataset/doc3d/"
    train_dataset = Doc3dDataset(root=data_path,split="train")
    for i, (images, labels) in enumerate(train_dataset):
        print((images, labels))

