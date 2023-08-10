from typing import Any, List

import os
import torch
import torchvision
from torchvision.utils import save_image
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from src.models.components.slotaug.slotaug_utils import *
from src.utils.evaluator import ARIEvaluator, mIoUEvaluator
from src.utils.vis_utils import visualize

from hydra.core.hydra_config import HydraConfig

# from torchmetrics.detection.mean_ap import MeanAveragePrecision

class LitSlotAugAutoEncoder(LightningModule):
    """LightningModule for SlotAugAutoEncoder.

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
        name: str = "slotaug",
        ori_only_epochs: int = 0,
        loss_sc_weight: float = 0.01,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.name = name
        self.ori_only_epochs = ori_only_epochs
        self.loss_sc_weight = loss_sc_weight

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.val_fg_ari_ori = ARIEvaluator()
        self.val_fg_ari_aug = ARIEvaluator()

        self.val_ari_ori = ARIEvaluator()
        self.val_ari_aug = ARIEvaluator()

        self.val_miou_ori = mIoUEvaluator()
        self.val_miou_aug = mIoUEvaluator()

        # for averaging loss across batches
        self.train_loss_ori = MeanMetric()
        self.val_loss_ori = MeanMetric()
        self.train_loss_aug = MeanMetric()
        self.val_loss_aug = MeanMetric()
        self.train_loss_img = MeanMetric()
        self.val_loss_img = MeanMetric()
        self.train_loss_sc = MeanMetric()
        self.val_loss_sc = MeanMetric()
        self.train_loss_total = MeanMetric()
        self.val_loss_total = MeanMetric()

    def forward(
        self, 
        img_ori: torch.Tensor, 
        img_aug: torch.Tensor, 
        insts_ori2aug: dict, 
        insts_aug2ori: dict, 
    ):
        outputs = self.net(img_ori, None, insts_ori2aug, insts_aug2ori)
        return outputs

    def model_step(self, batch: Any):
        img_ori = batch["img_ori"]
        img_aug = batch["img_aug"]
        insts_ori2aug = batch["insts_ori2aug"]
        insts_aug2ori = batch["insts_aug2ori"]

        outputs = self.forward(img_ori, img_aug, insts_ori2aug, insts_aug2ori)
        loss_ori = self.criterion(outputs["recon_combined_ori"], img_ori)
        loss_aug = self.criterion(outputs["recon_combined_ori2aug"], img_aug)
        loss_sc = self.criterion(outputs["slots_ori_revisited"], outputs["slots_ori"].detach()) # slot consistency

        outputs["attns_ori2aug"], outputs["normed_attns_ori2aug"] = masks_to_attns(outputs["masks_ori2aug"])
        # TODO: our convention is get obj_pos from alpha masks, but here is not the case
        obj_pos_aug = get_obj_pos_from_attns(outputs["normed_attns_aug"])
        obj_pos_ori2aug = get_obj_pos_from_attns(outputs["normed_attns_ori2aug"])

        obj_pos_filtered_ori2aug, valid_slots_mask_ori2aug  = filter_no_obj(obj_pos_ori2aug, outputs["attns_ori2aug"], replace_value=-2.0) # -2 for no_obj
        obj_pos_filtered_aug, valid_slots_mask_aug  = filter_no_obj(obj_pos_aug, outputs["attns_aug"], replace_value=-2.0) # -2 for no_obj
        obj_pos_filtered_ori2aug, background_mask_ori2aug  = filter_background(obj_pos_filtered_ori2aug, outputs["attns_ori2aug"], replace_value=2.0) # 2 for background
        obj_pos_filtered_aug, background_mask_aug  = filter_background(obj_pos_filtered_aug, outputs["attns_aug"], replace_value=2.0) # 2 for background

        # matched_idxs = matching_pairs(obj_pos_filtered_aug, obj_pos_filtered_ori2aug) # (B, K)
        # slots_aug = rearrange_by_matched_idx(outputs["slots_aug"], matched_idxs)
        # obj_pos_filtered_aug = rearrange_by_matched_idx(obj_pos_filtered_aug, matched_idxs)
        # valid_slots_mask_aug = rearrange_by_matched_idx(valid_slots_mask_aug, matched_idxs)
        
        # valid_slots_mask = (valid_slots_mask_ori2aug * valid_slots_mask_aug).unsqueeze(-1)
        # slots_ori2aug = outputs["slots_ori2aug"] * valid_slots_mask
        # slots_aug = slots_aug * valid_slots_mask

        return {"loss_ori": loss_ori, 
                "loss_aug": loss_aug,
                "loss_sc": loss_sc,
                "outputs": outputs}

    def on_train_start(self):
        pass 

    def training_step(self, batch: Any, batch_idx: int):
        model_step_outputs = self.model_step(batch)
        loss_ori = model_step_outputs["loss_ori"]
        loss_aug = model_step_outputs["loss_aug"]
        loss_sc = model_step_outputs["loss_sc"]
        outputs = model_step_outputs["outputs"]
        
        # recon loss
        if self.current_epoch < self.ori_only_epochs:
            loss_img = loss_ori
        else:
            loss_img = loss_ori + loss_aug
        # consistency loss
        loss = loss_img + \
               loss_sc * self.loss_sc_weight

        # update and log metrics
        self.train_loss_ori(loss_ori)
        self.log("train/loss_ori", self.train_loss_ori, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss_aug(loss_aug)
        self.log("train/loss_aug", self.train_loss_aug, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss_img(loss_img)
        self.log("train/loss_img", self.train_loss_img, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss_sc(loss_sc)
        self.log("train/loss_slot_consistency", self.train_loss_sc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_loss_total(loss)
        self.log("train/loss_total", self.train_loss_total, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def _shared_eval_step(self, batch: Any, batch_idx: int):

        model_step_outputs = self.model_step(batch)
        loss_ori = model_step_outputs["loss_ori"]
        loss_aug = model_step_outputs["loss_aug"]
        loss_sc = model_step_outputs["loss_sc"]
        outputs = model_step_outputs["outputs"]

        loss_img = loss_ori + loss_aug
        loss = loss_img + loss_sc
        
        # update and log metrics
        self.val_loss_ori(loss_ori)
        self.log("val/loss_ori", self.val_loss_ori, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_aug(loss_aug)
        self.log("val/loss_aug", self.val_loss_aug, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_img(loss_img)
        self.log("val/loss_img", self.val_loss_img, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_sc(loss_sc)
        self.log("val/loss_slot_consistency", self.val_loss_sc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_total(loss)
        self.log("val/loss_total", self.val_loss_total, on_step=False, on_epoch=True, prog_bar=True)

        self.val_fg_ari_ori.evaluate(outputs["masks_ori"].squeeze(-1), batch["masks_ori"][:, 1:].squeeze(-1))
        self.val_ari_ori.evaluate(outputs["masks_ori"].squeeze(-1), batch["masks_ori"].squeeze(-1))
        self.val_miou_ori.evaluate(outputs["masks_ori"].squeeze(-1), batch["masks_ori"].squeeze(-1))

        self.val_fg_ari_aug.evaluate(outputs["masks_ori2aug"].squeeze(-1), batch["masks_aug"][:, 1:].squeeze(-1))
        self.val_ari_aug.evaluate(outputs["masks_ori2aug"].squeeze(-1), batch["masks_aug"].squeeze(-1))
        self.val_miou_aug.evaluate(outputs["masks_ori2aug"].squeeze(-1), batch["masks_aug"].squeeze(-1))

        return loss, outputs

    def validation_step(self, batch: Any, batch_idx: int):
        loss, outputs = self._shared_eval_step(batch, batch_idx)

        # visualization
        if batch_idx == 0:
            n_sampels = 4
            wandb_img_list = list()
            for vis_idx in range(n_sampels):
                vis_ori = visualize(
                    image=batch["img_ori"][vis_idx].unsqueeze(0),
                    recon_combined=outputs["recon_combined_ori"][vis_idx].unsqueeze(0),
                    recons=outputs["recons_ori"][vis_idx].unsqueeze(0),
                    pred_masks=outputs["masks_ori"][vis_idx].unsqueeze(0),
                    gt_masks=batch["masks_ori"][vis_idx].unsqueeze(0),
                    attns=outputs["attns"][vis_idx].unsqueeze(0),
                    colored_box=True,
                )
                vis_aug = visualize(
                    image=batch["img_aug"][vis_idx].unsqueeze(0),
                    recon_combined=outputs["recon_combined_ori2aug"][vis_idx].unsqueeze(0),
                    recons=outputs["recons_ori2aug"][vis_idx].unsqueeze(0),
                    pred_masks=outputs["masks_ori2aug"][vis_idx].unsqueeze(0),
                    gt_masks=batch["masks_aug"][vis_idx].unsqueeze(0),
                    attns=outputs["attns"][vis_idx].unsqueeze(0),
                    colored_box=True,
                )

                grid_ori = torchvision.utils.make_grid(vis_ori, nrow=1, pad_value=0)
                grid_aug = torchvision.utils.make_grid(vis_aug, nrow=1, pad_value=0)
                grid = torch.cat([grid_ori, grid_aug], dim=1) # concat along height

                wandb_img = wandb.Image(grid, caption=f"{self.name}/{self.current_epoch+1}e")
                wandb_img_list.append(wandb_img)
                
            self.logger.log_image(key="Visualization on Validation Set", images=wandb_img_list)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        val_fg_ari_ori = self.val_fg_ari_ori.get_results()
        self.val_fg_ari_ori.reset()
        val_fg_ari_aug = self.val_fg_ari_aug.get_results()
        self.val_fg_ari_aug.reset()

        val_ari_ori = self.val_ari_ori.get_results()
        self.val_ari_ori.reset()
        val_ari_aug = self.val_ari_aug.get_results()
        self.val_ari_aug.reset()

        val_miou_ori = self.val_miou_ori.get_results()
        self.val_miou_ori.reset()
        val_miou_aug = self.val_miou_aug.get_results()
        self.val_miou_aug.reset()

        self.log_dict(
            {
                "val/fg-ari_ori": val_fg_ari_ori,
                "val/ari_ori": val_ari_ori,
                "val/miou_ori": val_miou_ori,
                "val/fg-ari_aug": val_fg_ari_aug,
                "val/ari_aug": val_ari_aug,
                "val/miou_aug": val_miou_aug,
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
    _ = LitSlotAugAutoEncoder(None, None, None)
