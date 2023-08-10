from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from src.models.components.slotaug.slotaug import SlotAug
from src.models.components.slota.slota_utils import *
import src.models.components.slota.models_vit as models_vit
from src.models.components.slota.vit_utils import *

class SlotAugAutoEncoder(nn.Module):
    """Builds Slot Attention-based auto-encoder for object discovery.

    Args:
        num_slots (int): Number of slots in Slot Attention.
    """

    def __init__(
        self,
        img_size: int = 128,
        num_slots: int = 7,
        num_iter: int = 3,
        num_iter_insts: int = 1,
        num_attn_heads: int = 1,
        hid_dim: int = 64,
        slot_dim: int = 64,
        mlp_hid_dim: int = 128,
        eps: float = 1e-8,
        enc_depth: int = 4,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
        aux_identity: bool = True,
        use_pe: bool = True,
        enc_type: str = "cnn",
        enc_norm_type: str = "bn",
        enc_ckpt: str = "", # only for pretrained vit
        freeze_vit: bool = True, # only for vit,
        dec_type: str = "sb",
        query_dim: int = 64, # only for srt_slotwise
        num_split: int = 4, # only for srt_slotwise
        ray_size: int = 128, # only for srt_slotwise
        sigmoid: bool = True, # only for srt_slotwise
    ):
        super().__init__()
        self.num_slots = num_slots
        self.aux_identity = aux_identity
        self.enc_type = enc_type
        self.dec_type = dec_type
        self.sigmoid = sigmoid
        if enc_type.lower() == "cnn":
            self.encoder = Encoder(
                img_size=img_size,
                hid_dim=hid_dim,
                enc_depth=enc_depth,
                use_pe=use_pe,
            )
        elif "resnet" in enc_type.lower():
            if enc_type.lower() == "resnet34":
                enc_depth = 34
                hid_dim = 512
            elif enc_type.lower() == "resnet50":
                enc_depth = 50
                hid_dim = 2048
                
            self.encoder = ResNetEncoder(
                feat_size=img_size//8,
                hid_dim=hid_dim,
                enc_depth=enc_depth,
                use_pe=use_pe,
                norm_type=enc_norm_type,
            )
        elif "vit" in enc_type.lower():
            if "small" in enc_type.lower():
                hid_dim = 384
            elif "base" in enc_type.lower():
                hid_dim = 768
            elif "large" in enc_type.lower():
                hid_dim = 1024
            elif "huge" in enc_type.lower():
                hid_dim = 1280

            self.encoder = models_vit.__dict__[enc_type](
                drop_path_rate=0.1,
                img_size=img_size,
            )

            if enc_ckpt != "":
                checkpoint = torch.load(enc_ckpt, map_location='cpu')

                print("Load pre-trained checkpoint from: %s" % enc_ckpt)
                checkpoint_model = checkpoint['model']
                state_dict = self.encoder.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(self.encoder, checkpoint_model)

                # load pre-trained model
                msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
                print(msg)

                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

                # manually initialize fc layer
                trunc_normal_(self.encoder.head.weight, std=2e-5)

                # freeze encoder
                if freeze_vit:
                    for param in self.encoder.parameters(): 
                        param.requires_grad = False 

        if dec_type == 'sb':
            self.decoder = Decoder(
                img_size=img_size,
                slot_dim=slot_dim,
                dec_hid_dim=dec_hid_dim,
                dec_init_size=dec_init_size,
                dec_depth=dec_depth,
            )
        elif dec_type == 'srt':
            self.decoder = SRTDecoder(
            img_size=img_size,
            slot_dim=slot_dim,
            dec_hid_dim=dec_hid_dim,
            dec_init_size=dec_init_size,
            dec_depth=dec_depth,
        )
        elif dec_type == 'srt_slotwise':
            self.decoder = SRTDecoder_Slotwise(
            img_size=img_size,
            slot_dim=slot_dim,
            dec_hid_dim=dec_hid_dim,
            query_dim=query_dim,
            num_split=num_split,
            ray_size=ray_size
        )

        self.slotaug = SlotAug(
            num_slots=num_slots,
            num_iter=num_iter,
            num_iter_insts=num_iter_insts,
            num_attn_heads=num_attn_heads,
            slot_dim=slot_dim,
            hid_dim=hid_dim,
            mlp_hid_dim=mlp_hid_dim,
            aux_identity=self.aux_identity,
            eps=eps,
        )
                        
        self.num_iter = num_iter
        self.num_iter_insts = num_iter_insts

    def forward(self, img_ori, img_aug, insts_ori2aug, insts_aug2ori):
        # `img_ori`: (B, C, H, W)
        # `img_aug`: (B, C, H, W)
        # `insts`: (B, K, 11) obj_pos (2) + rotate (1) + translate (2) + scale (1) + color (3) + flip (2)
        
        B, C, H, W = img_ori.shape

        # Convolutional encoder with position embedding
        x = self.encoder(img_ori)  # CNN Backbone
        if img_aug != None:
            x_aug = self.encoder(img_aug)  # CNN Backbone
        else:
            x_aug = None
        # `x`: (B, height * width, hid_dim)

        # Slot Attention module.
        slotaug_outputs = self.slotaug(inputs_ori=x, inputs_aug=x_aug, insts_ori2aug=insts_ori2aug, insts_aug2ori=insts_aug2ori)
        slots_ori = slotaug_outputs["slots_ori"]
        slots_ori2aug = slotaug_outputs["slots_ori2aug"]
        # `slots`: (N, K, slot_dim)

        x_ori = self.decoder(slots_ori) # sb: (B*K, height, width, 4)
        x_ori2aug = self.decoder(slots_ori2aug) # sb: (B*K, height, width, 4)
        # `x`: (B*K, height, width, num_channels+1)

        if self.dec_type == 'sb':
            # Undo combination of slot and batch dimension; split alpha masks
            recons_ori, masks_ori = x_ori.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
            recons_ori2aug, masks_ori2aug = x_ori2aug.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
            # `recons`: (B, K, height, width, num_channels)
            # `masks`: (B, K, height, width, 1)

            # Normalize alpha masks over slots.
            masks_ori = nn.Softmax(dim=1)(masks_ori)
            masks_ori2aug = nn.Softmax(dim=1)(masks_ori2aug)

            recon_combined_ori = torch.sum(recons_ori * masks_ori, dim=1)  # Recombine image
            recon_combined_ori2aug = torch.sum(recons_ori2aug * masks_ori2aug, dim=1)  # Recombine image

        # TODO: sanity check for srt decoder along with all encoder types
        elif self.dec_type == 'srt':
            recons_ori = torch.zeros(B, self.num_slots, H, W, C).to(x_ori.device) # no individual recon per slot in unidec
            recons_ori2aug = torch.zeros(B, self.num_slots, H, W, C).to(x_ori.device) # no individual recon per slot in unidec
            
            # use attn_masks as segm masks
            attns_masks = slotaug_outputs["attns_ori"].clone() # (B, K, N_heads, N_in) (N_in = H * W)
            if "resnet" in self.enc_type or "vit" in self.enc_type:
                upsample_trans = nn.Upsample(size=(H, W), mode='nearest')
                # HACK (16, 16) is now fixed 
                masks_ori = upsample_trans(torch.mean(attns_masks, dim=2).reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, H, W)[..., None] # (B, K, N_in, 1))
            elif self.enc_type == "cnn":
                masks_ori = torch.mean(attns_masks, dim=2).reshape(B, self.num_slots, H, W)[..., None] # (B, K, N_in, 1)
            
            masks_ori2aug = torch.zeros(B, self.num_slots, H, W, 1).to(x_ori.device) # no alpha mask in unidec

            recon_combined_ori = x_ori.reshape(B, H, W, C)
            recon_combined_ori2aug = x_ori2aug.reshape(B, H, W, C)
        

        # TODO: implementation for srt_slotwise decoder
        elif self.dec_type == 'srt_slotwise':
            # individual recon per slot
            # Undo combination of slot and batch dimension; split alpha masks
            recons_ori, masks_ori = x_ori.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
            recons_ori2aug, masks_ori2aug = x_ori2aug.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
            # `recons`: (B, K, height, width, num_channels)
            # `masks`: (B, K, height, width, 1)

            if self.sigmoid:
                recons_ori = torch.sigmoid(recons_ori)
                recons_ori2aug = torch.sigmoid(recons_ori2aug)

            # Normalize alpha masks over slots.
            masks_ori = nn.Softmax(dim=1)(masks_ori)
            masks_ori2aug = nn.Softmax(dim=1)(masks_ori2aug)

            recon_combined_ori = torch.sum(recons_ori * masks_ori, dim=1)  # Recombine image
            recon_combined_ori2aug = torch.sum(recons_ori2aug * masks_ori2aug, dim=1)  # Recombine image


        recon_combined_ori = recon_combined_ori.permute(0, 3, 1, 2)
        recon_combined_ori2aug = recon_combined_ori2aug.permute(0, 3, 1, 2)
        # `recon_combined`: (batch_size, num_channels, height, width)

        outputs = dict()
        outputs["recon_combined_ori"] = recon_combined_ori
        outputs["recon_combined_ori2aug"] = recon_combined_ori2aug 
        outputs["recons_ori"] = recons_ori
        outputs["recons_ori2aug"] = recons_ori2aug
        outputs["masks_ori"] = masks_ori
        outputs["masks_ori2aug"] = masks_ori2aug
        # outputs["slots"] = slots
        outputs["slots_ori"] = slots_ori
        outputs["slots_ori2aug"] = slots_ori2aug
        outputs["slots_ori_revisited"] = slotaug_outputs["slots_ori_revisited"]
        outputs["normed_attns"] = slotaug_outputs["normed_attns"]
        outputs["normed_attns_ori"] = slotaug_outputs["normed_attns_ori"]
        
        outputs["slots_aug"] = slotaug_outputs["slots_aug"]
        outputs["normed_attns_aug"] = slotaug_outputs["normed_attns_aug"]

        if "resnet" in self.enc_type or "vit" in self.enc_type:
            # for resnet 16 16 attns
            upsample_trans = nn.Upsample(size=(H, W), mode='nearest')
            outputs["attns"] = upsample_trans(slotaug_outputs["attns"].reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, 1, H*W)
            outputs["attns_ori"] = upsample_trans(slotaug_outputs["attns_ori"].reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, 1, H*W)
            outputs["attns_aug"] = upsample_trans(slotaug_outputs["attns_aug"].reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, 1, H*W)
        else:
            outputs["attns"] = slotaug_outputs["attns"]
            outputs["attns_ori"] = slotaug_outputs["attns_ori"]
            outputs["attns_aug"] = slotaug_outputs["attns_aug"]


        

        return outputs


if __name__ == "__main__":
    _ = SlotAugAutoEncoder()
