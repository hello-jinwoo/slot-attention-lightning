import json
import os
from collections import defaultdict
import random
import math

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms

from src.data.components.data_utils import * # https://github.com/limacv/RGB_HSV_HSL

class TETRO(Dataset):
    def __init__(
        self,
        data_dir: str = "data/tetrominoes",
        img_size: int = 64,
        crop_size: int = 80,
        template_size: int = 35,
        padding_size: int = 28,
        transform_base: transforms.Compose = None,
        train: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.crop_size = crop_size
        self.template_size = template_size
        self.train = train
        self.stage = "train" if train else "val"

        self.max_num_objs = 4
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")
        self.metadata = json.load(
            open(os.path.join(self.scene_dir, f"TETROMINOES_{self.stage}_scenes.json"))
        )

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

        self.transform_background = transforms.Pad(padding=(padding_size,padding_size), fill=1, padding_mode='constant')
        self.transform_base = transform_base
        self.transform_resize = transforms.Resize(size=img_size)

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
        img = self.transform_base(img)
        img = self.transform_resize(img)
        _, H, W = img.shape


        sample = {"image": img}

        if not self.train:
            masks = list()
            for i, mask_filename in enumerate(self.masks[filename]):
                mask = (
                    read_image(os.path.join(self.mask_dir, mask_filename), ImageReadMode.GRAY)
                    .div(255)
                    .long()
                )

                #background masking
                if i == 0:
                    mask = self.transform_background(mask)
                else:
                    mask = self.transform_base(mask)
                masks.append(self.transform_resize(mask))
            masks = torch.cat(masks, dim=0).unsqueeze(-1)
            # `masks`: (num_objects + 1, 3, H, W)

            num_masks = masks.shape[0]
            if num_masks < self.max_num_masks:
                pad = torch.zeros((self.max_num_masks - num_masks, self.img_size, self.img_size, 1))
                masks = torch.cat((masks, pad), dim=0)
            # `masks`: (max_num_masks, H, W, 1)

            sample["masks"] = masks.float()
            sample["num_objects"] = num_masks - 1
            
        return sample
        # `insts`: (K, 9)
        # `img_ori`: (3, H, W)
        # `img_aug`: (3, H, W)
        # `masks_ori`: (max_num_masks, H, W, 1)
        # `masks_aug`: (max_num_masks, H, W, 1)
        # `num_objects`: int

    def __len__(self):
        return self.num_files