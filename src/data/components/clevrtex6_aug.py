import json
import os
from collections import defaultdict
import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms

from src.data.components.data_utils import *

class ClevrTex6Aug(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevrtex6",
        img_size: int = 128,
        crop_size: int = 196,
        template_size: int = 240,
        transform_contents: str = 'translate',
        random_swap: bool = False, 
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.crop_size = crop_size
        self.template_size = template_size
        self.train = train
        self.stage = "train" if train else "val"

        self.max_num_objs = 6
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform_contents = transform_contents.split(",")
        self.random_swap = random_swap

    def __getitem__(self, index):
        image_filename = self.files[index]
        img = (
            read_image(os.path.join(self.image_dir, image_filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        transform_content = random.choice(self.transform_contents)
        insts_ori2aug, transform_ori, transform_aug = get_transforms(
            transform_content=transform_content,
            img_size=self.img_size, 
            crop_size=self.crop_size, 
            template_size=self.template_size, 
            max_num_masks=self.max_num_masks,
        )
        img_ori = transform_ori(img) # (3, H, W)
        img_aug = transform_aug(img) # (3, H, W)

        if 'color' in transform_content:
            img_aug, insts_ori2aug = color_transform(img_aug, insts_ori2aug)
        insts_aug2ori = get_inv_insts(insts_ori2aug)

        sample = {
            "insts_ori2aug": insts_ori2aug,
            "insts_aug2ori": insts_aug2ori,
            "img_ori": img_ori, 
            "img_aug": img_aug
        }

        scene_name = image_filename[:-3] + "json"
        metadata = json.load(open(os.path.join(self.scene_dir, self.stage, scene_name)))

        if not self.train:
            mask_filename = image_filename[:-4] + "_flat.png"
            masks = read_image(os.path.join(self.mask_dir, mask_filename)).long().squeeze(0)
            masks = F.one_hot(masks, self.max_num_masks).permute(2, 0, 1)
            masks_ori = transform_ori(masks).unsqueeze(-1)
            masks_aug = transform_aug(masks).unsqueeze(-1)
            # `masks`: (max_num_masks, H, W, 1)

            sample["masks_ori"] = masks_ori.float()
            sample["masks_aug"] = masks_aug.float()
            sample["num_objects"] = len(metadata["objects"])

        if self.random_swap: 
            if torch.randn(1).item() < 0.5:
                sample["img_ori"], sample["img_aug"] = sample["img_aug"], sample["img_ori"]
                sample["insts_ori2aug"], sample["insts_aug2ori"] = sample["insts_aug2ori"], sample["insts_ori2aug"]

                if not self.train:
                    sample["masks_ori"], sample["masks_aug"] = sample["masks_aug"], sample["masks_ori"]
                    
        return sample
        # `insts`: (K, 11)
        # `img_ori`: (3, H, W)
        # `img_aug`: (3, H, W)
        # `masks_ori`: (max_num_masks, H, W, 1)
        # `masks_aug`: (max_num_masks, H, W, 1)
        # `num_objects`: int

    def __len__(self):
        return self.num_files