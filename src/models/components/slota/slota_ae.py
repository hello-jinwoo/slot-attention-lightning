from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from src.models.components.slota.slota import SlotAttention
from src.models.components.slota.slota_utils import *
import src.models.components.slota.models_vit as models_vit
from src.models.components.slota.vit_utils import *

class SlotAttentionAutoEncoder(nn.Module):
    """Builds Slot Attention-based auto-encoder for object discovery.

    Args:
        num_slots (int): Number of slots in Slot Attention.
    """

    def __init__(
        self,
        img_size: int = 128,
        num_slots: int = 7,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        hid_dim: int = 64,
        slot_dim: int = 64,
        mlp_hid_dim: int = 128,
        eps: float = 1e-8,
        enc_depth: int = 4,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
        ark_size: int = 5,
        enc_type: str = "cnn",
        enc_norm_type: str = "bn",
        enc_ckpt: str = "", # only for pretrained vit
        freeze_vit: bool = True, # only for vit,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.enc_type = enc_type

        # TODO: change encoder_cnn to encoder (now, to preserve the sync with previous method, we maintain it as encoder_cnn)
        if enc_type.lower() == "cnn":
            self.encoder_cnn = Encoder(
                img_size=img_size,
                hid_dim=hid_dim,
                enc_depth=enc_depth,
            )
        elif "resnet" in enc_type.lower():
            if enc_type.lower() == "resnet34":
                enc_depth = 34
                hid_dim = 512
            elif enc_type.lower() == "resnet50":
                enc_depth = 50
                hid_dim = 2048
                
            self.encoder_cnn = ResNetEncoder(
                feat_size=img_size//8,
                hid_dim=hid_dim,
                enc_depth=enc_depth,
                norm_type=enc_norm_type,
            )
        elif "vit" in enc_type.lower():
            if "tiny" in enc_type.lower():
                pass # TODO:
            elif "small" in enc_type.lower():
                hid_dim = 384
            elif "base" in enc_type.lower():
                hid_dim = 768
            elif "large" in enc_type.lower():
                hid_dim = 1024
            elif "huge" in enc_type.lower():
                hid_dim = 1280

            self.encoder_cnn = models_vit.__dict__[enc_type](
                drop_path_rate=0.1,
                img_size=img_size,
            )

            if enc_ckpt != "":
                checkpoint = torch.load(enc_ckpt, map_location='cpu')

                print("Load pre-trained checkpoint from: %s" % enc_ckpt)
                checkpoint_model = checkpoint['model']
                state_dict = self.encoder_cnn.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]

                # interpolate position embedding
                interpolate_pos_embed(self.encoder_cnn, checkpoint_model)

                # load pre-trained model
                msg = self.encoder.load_state_dict(checkpoint_model, strict=False)
                print(msg)

                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

                # manually initialize fc layer
                trunc_normal_(self.encoder_cnn.head.weight, std=2e-5)

                # freeze encoder
                if freeze_vit:
                    for param in self.encoder.parameters(): 
                        param.requires_grad = False 

        self.decoder_cnn = Decoder(
            img_size=img_size,
            slot_dim=slot_dim,
            dec_hid_dim=dec_hid_dim,
            dec_init_size=dec_init_size,
            dec_depth=dec_depth,
        )

        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            num_iterations=num_iterations,
            num_attn_heads=num_attn_heads,
            slot_dim=slot_dim,
            hid_dim=hid_dim,
            mlp_hid_dim=mlp_hid_dim,
            ark_size=ark_size,
            eps=eps,
        )

    def forward(self, image):
        # `image`: (batch_size, num_channels, height, width)
        B, C, H, W = image.shape

        # Convolutional encoder with position embedding
        x = self.encoder_cnn(image)  # CNN Backbone
        # `x`: (B, height * width, hid_dim)

        # Slot Attention module.
        slota_outputs = self.slot_attention(x)
        slots = slota_outputs["slots"]
        # `slots`: (N, K, slot_dim)

        x = self.decoder_cnn(slots)
        # `x`: (B*K, height, width, num_channels+1)

        # Undo combination of slot and batch dimension; split alpha masks
        recons, masks = x.reshape(B, self.num_slots, H, W, C + 1).split([3, 1], dim=-1)
        # `recons`: (B, K, height, width, num_channels)
        # `masks`: (B, K, height, width, 1)

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)

        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined`: (batch_size, num_channels, height, width)

        outputs = dict()
        outputs["recon_combined"] = recon_combined
        outputs["recons"] = recons
        outputs["masks"] = masks
        outputs["slots"] = slots
        outputs["attns"] = slota_outputs["attns"]
        outputs["normed_attns"] = slota_outputs["normed_attns"]
        # `attns`: (B, K, N_heads, N_in), after softmax before normalization
        # `normed_attns`: (B, K, N_heads, N_in), after softmax and normalization

        if "resnet" in self.enc_type or "vit" in self.enc_type:
            # HACK: here, the encoding feature map size is hard coded with (16, 16)
            upsample_trans = nn.Upsample(size=(H, W), mode='nearest')
            outputs["attns"] = upsample_trans(slota_outputs["attns"].reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, 1, H*W)
            outputs["normed_attns"] = upsample_trans(slota_outputs["normed_attns"].reshape(B, self.num_slots, 16, 16)).reshape(B, self.num_slots, 1, H*W)
        else:
            outputs["attns"] = slota_outputs["attns"]
            outputs["normed_attns"] = slota_outputs["normed_attns"]
            

        return outputs


if __name__ == "__main__":
    _ = SlotAttentionAutoEncoder()
