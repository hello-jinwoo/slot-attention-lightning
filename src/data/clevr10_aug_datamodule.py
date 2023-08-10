from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from src.data.components.clevr10_aug import CLEVR10Aug


class CLEVR10AugDataModule(LightningDataModule):
    """LightningDataModule for CLEVR6 with DataAugmentation.

    A DataModule implements 5 key methods:

        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        name: str = "clv10_aug",
        data_dir: str = "data/",
        num_patches: int = 1,
        img_size: int = 128,
        crop_size: int = 192,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        transform_contents: str = 'translate',
        random_swap: bool = False, 
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # transforms applied to the original image
        self.transform_contents = transform_contents

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        self.data_train = CLEVR10Aug(
            data_dir=self.hparams.data_dir,
            img_size=self.hparams.img_size,
            crop_size=self.hparams.crop_size,
            transform_contents=self.transform_contents,
            random_swap=self.hparams.random_swap, 
            train=True,
        )

        self.data_val = CLEVR10Aug(
            data_dir=self.hparams.data_dir,
            img_size=self.hparams.img_size,
            crop_size=self.hparams.crop_size,
            transform_contents=self.transform_contents,
            random_swap=self.hparams.random_swap, 
            train=False,
        )

        self.data_test = self.data_val 

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = CLEVR10AugDataModule()