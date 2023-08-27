import json
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms


class PTR(Dataset):
    def __init__(
        self,
        data_dir: str = "data/PTR",
        img_size: int = 128,
        transform: transforms.Compose = None,
        train: bool = True,
        num_samples: int = 0,
    ):
        super().__init__()

        self.max_num_objs = 6
        self.max_num_masks = self.max_num_objs + 1

        self.img_size = img_size
        self.train = train
        self.stage = "train" if train else "val"

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")

        self.files = sorted(os.listdir(self.image_dir))
        if num_samples > 0: # sample N images
            random.seed(707)
            random.shuffle(self.files)
            self.files = self.files[:num_samples]
        self.num_files = len(self.files)

        self.transform = transform

        if not train:
            self.masks = defaultdict(list)
            masks = sorted(os.listdir(self.mask_dir))
            for mask in masks:
                split = mask.split("_")
                filename = "_".join(split[:3]) + ".png"
                self.masks[filename].append(mask)
            del masks

    def __getitem__(self, index):
        filename = self.files[index]

        img = (
            read_image(os.path.join(self.image_dir, filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )
        img = self.transform(img)
        sample = {"image": img}

        if not self.train:
            masks = list()
            for mask_filename in self.masks[filename]:
                mask = (
                    read_image(os.path.join(self.mask_dir, mask_filename), ImageReadMode.GRAY)
                    .div(255)
                    .long()
                )
                mask = self.transform(mask)
                masks.append(mask)
            masks = torch.cat(masks, dim=0).unsqueeze(-1)
            # `masks`: (num_objects + 1, H, W, 1)

            num_masks = masks.shape[0]
            if num_masks < self.max_num_masks:
                masks = torch.cat(
                    (
                        masks,
                        torch.zeros(
                            (self.max_num_masks - num_masks, self.img_size, self.img_size, 1)
                        ),
                    ),
                    dim=0,
                )
            # `masks``: (max_num_masks, H, W, 1)

            sample["masks"] = masks.float()
            sample["num_objects"] = num_masks - 1

        return sample

    def __len__(self):
        return self.num_files