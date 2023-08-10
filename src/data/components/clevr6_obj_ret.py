import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms as transforms

from src.data.components.data_utils import *

class CLEVR6Obj(Dataset):
    def __init__(
        self,
        data_dir: str = "data/clevr_with_masks/CLEVR6",
        img_size: int = 128,
        crop_size: int = 196,
        train: bool = True,
        size: bool = False,
    ):
        super().__init__()

        self.img_size = img_size
        self.crop_size = crop_size
        self.train = train
        self.stage = "train" if train else "val"
        self.size = size
        self.max_num_objs = 6
        self.max_num_masks = self.max_num_objs + 1

        self.image_dir = os.path.join(data_dir, "images", self.stage)
        self.mask_dir = os.path.join(data_dir, "masks", self.stage)
        self.scene_dir = os.path.join(data_dir, "scenes")
        self.metadata = json.load(
            open(os.path.join(self.scene_dir, f"CLEVR_{self.stage}_scenes.json"))
        )

        self.files = sorted(os.listdir(self.image_dir))
        self.num_files = len(self.files)

    
    def __getitem__(self, index):
        filename = self.metadata["scenes"][index]["image_filename"]
        objects = self.metadata["scenes"][index]["objects"]
        
        if self.size:
            object_property = torch.stack([torch.tensor([obj['material'], obj['shape'], obj['color'], obj['size']]) for obj in objects])
        else:
            object_property = torch.stack([torch.tensor([obj['material'], obj['shape'], obj['color']]) for obj in objects])

        img = (
            read_image(os.path.join(self.image_dir, filename), ImageReadMode.RGB)
            .float()
            .div(255.0)
        )

        transform = transforms.Compose([
            transforms.CenterCrop((self.crop_size, self.crop_size)),
            transforms.Resize((self.img_size, self.img_size)),
        ])

        img_ori = transform(img) # (3, H, W)

        sample = {
            "img_ori": img_ori, 
            "property": object_property,
        }
            
        return sample
        # `insts_ori2aug`: (K, 9)
        # `insts_aug2ori`: (K, 9)
        # `img_ori`: (3, H, W)
        # `img_aug`: (3, H, W)
        # `masks_ori`: (max_num_masks, H, W, 1)
        # `masks_aug`: (max_num_masks, H, W, 1)
        # `num_objects`: int

    def __len__(self):
        return self.num_files