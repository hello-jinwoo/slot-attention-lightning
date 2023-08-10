import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms

from src.data.components.data_utils import *

class CLEVR10Aug(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevr_with_masks/CLEVR10",
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

        self.max_num_objs = 10
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")
        self.metadata = json.load(
            open(os.path.join(self.scene_dir, f"CLEVR_{self.stage}_scenes.json"))
        )

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform_contents = transform_contents.split(",")
        self.random_swap = random_swap

        if not train:
            self.masks = defaultdict(list)
            masks = sorted(os.listdir(self.mask_dir))
            for mask in masks:
                split = mask.split("_")
                filename = "_".join(split[:3]) + ".png"
                self.masks[filename].append(mask)
            del masks
    
    def __getitem__(self, index):
        filename = self.metadata["scenes"][index]["image_filename"]
        img = (
            read_image(os.path.join(self.image_dir, filename), ImageReadMode.RGB)
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

        if not self.train:
            masks = list()
            masks_ori = list() 
            masks_aug = list() 
            for mask_filename in self.masks[filename]:
                mask = (
                    read_image(os.path.join(self.mask_dir, mask_filename), ImageReadMode.GRAY)
                    .div(255)
                    .long()
                )
                masks_ori.append(transform_ori(mask))
                masks_aug.append(transform_aug(mask))
            masks_ori = torch.cat(masks_ori, dim=0).unsqueeze(-1)
            masks_aug = torch.cat(masks_aug, dim=0).unsqueeze(-1)
            # `masks`: (num_objects + 1, 3, H, W)

            num_masks = masks_ori.shape[0]
            if num_masks < self.max_num_masks:
                pad = torch.zeros((self.max_num_masks - num_masks, self.img_size, self.img_size, 1))
                masks_ori = torch.cat((masks_ori, pad), dim=0)
                masks_aug = torch.cat((masks_aug, pad), dim=0)
            # `masks`: (max_num_masks, H, W, 1)

            sample["masks_ori"] = masks_ori.float()
            sample["masks_aug"] = masks_aug.float()
            sample["num_objects"] = num_masks - 1
        
        if self.random_swap: 
            if torch.randn(1).item() < 0.5:
                sample["img_ori"], sample["img_aug"] = sample["img_aug"], sample["img_ori"]
                sample["insts_ori2aug"], sample["insts_aug2ori"] = sample["insts_aug2ori"], sample["insts_ori2aug"]

                if not self.train:
                    sample["masks_ori"], sample["masks_aug"] = sample["masks_aug"], sample["masks_ori"]
            
        return sample
        # `insts_ori2aug`: (K, 8)
        # `insts_aug2ori`: (K, 8)
        # `img_ori`: (3, H, W)
        # `img_aug`: (3, H, W)
        # `masks_ori`: (max_num_masks, H, W, 1)
        # `masks_aug`: (max_num_masks, H, W, 1)
        # `num_objects`: int

    def __len__(self):
        return self.num_files