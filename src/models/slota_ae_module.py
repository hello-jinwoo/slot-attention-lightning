from typing import Any, List

import os
import torch
import torchvision
from torchvision.utils import save_image
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from utils.evaluator import ARIEvaluator, mIoUEvaluator
from utils.vis_utils import visualize

from hydra.core.hydra_config import HydraConfig

class LitSlotAttentionAutoEncoder(LightningModule):
    """LightningModule for SlotAttentionAutoEncoder.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: DictConfig,  # torch.optim.lr_scheduler,
        name: str = "slota",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_fg_ari = ARIEvaluator()
        self.val_fg_ari = ARIEvaluator()

        self.train_ari = ARIEvaluator()
        self.val_ari = ARIEvaluator()

        self.train_miou = mIoUEvaluator()
        self.val_miou = mIoUEvaluator()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor):
        outputs = self.net(x)
        return outputs

    def on_train_start(self):
        pass

    def model_step(self, batch: Any):
        img = batch["image"]
        outputs = self.forward(img)
        loss = self.criterion(outputs["recon_combined"], img)
        return loss, outputs

    def training_step(self, batch: Any, batch_idx: int):
        loss, outputs = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def _shared_eval_step(self, batch: Any, batch_idx: int):
        loss, outputs = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_fg_ari.evaluate(outputs["masks"].squeeze(-1), batch["masks"][:, 1:].squeeze(-1))
        self.val_ari.evaluate(outputs["masks"].squeeze(-1), batch["masks"].squeeze(-1))
        self.val_miou.evaluate(outputs["masks"].squeeze(-1), batch["masks"].squeeze(-1))

        return loss, outputs 
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, outputs = self._shared_eval_step(batch, batch_idx)

        if batch_idx == 0:
            n_sampels = 4
            wandb_img_list = list()
            for vis_idx in range(n_sampels):
                vis = visualize(
                    image=batch["image"][vis_idx].unsqueeze(0),
                    recon_combined=outputs["recon_combined"][vis_idx].unsqueeze(0),
                    recons=outputs["recons"][vis_idx].unsqueeze(0),
                    pred_masks=outputs["masks"][vis_idx].unsqueeze(0),
                    gt_masks=batch["masks"][vis_idx].unsqueeze(0),
                    attns=outputs["normed_attns"][vis_idx].unsqueeze(0),
                    colored_box=True,
                )
                grid = torchvision.utils.make_grid(vis, nrow=1, pad_value=0)
                wandb_img = wandb.Image(grid, caption=f"Epoch: {self.current_epoch}")
                wandb_img_list.append(wandb_img)
            self.logger.log_image(key="Visualization on Validation Set", images=wandb_img_list)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_fg_ari = self.val_fg_ari.get_results()
        self.val_fg_ari.reset()

        val_ari = self.val_ari.get_results()
        self.val_ari.reset()

        val_miou = self.val_miou.get_results()
        self.val_miou.reset()

        self.log_dict(
            {
                "val/fg-ari": val_fg_ari,
                "val/ari": val_ari,
                "val/miou": val_miou,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:

            def lr_lambda(step):

                if step < self.hparams.scheduler.warmup_steps:
                    warmup_factor = float(step) / float(
                        max(1.0, self.hparams.scheduler.warmup_steps)
                    )
                else:
                    warmup_factor = 1.0

                decay_factor = self.hparams.scheduler.decay_rate ** (
                    step / self.hparams.scheduler.decay_steps
                )

                return warmup_factor * decay_factor

            scheduler = self.hparams.scheduler.scheduler(
                optimizer=optimizer,
                lr_lambda=lr_lambda,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitSlotAttentionAutoEncoder(None, None, None)
