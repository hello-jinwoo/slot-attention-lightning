from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_

from src.models.components.sysbinder.sysbinder import SysBinder
from src.models.components.sysbinder.sysbinder_utils import *

class SysBinderAutoEncoder(nn.Module):
    """Builds Slot Attention-based auto-encoder for object discovery.

    Args:
        num_slots (int): Number of slots in Slot Attention.
    """

    def __init__(
        self,
        img_size: int = 128,
        img_dim: int = 3,
        num_slots: int = 7,
        num_iterations: int = 3,
        hid_dim: int = 64,
        slot_dim: int = 64,
        mlp_hid_dim: int = 128,
        num_prototypes: int = 64,
        vocab_size: int = 4096,
        d_model: int = 192,
        num_blocks: int = 8,
        num_decoder_heads: int = 4,
        num_decoder_layers: int = 8,
        dropout: float = 0.1,
        tau_start: float = 1.0,
        tau_final: float = 0.1,
        tau_steps: int = 30000,
    ):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.cnn_hidden_size = hid_dim
        self.slot_size = slot_dim
        self.mlp_hidden_size = mlp_hid_dim
        self.num_prototypes = num_prototypes
        self.image_channels = img_dim
        self.image_size = img_size
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_decoder_heads = num_decoder_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout

        # dvae
        self.dvae = dVAE(self.vocab_size, self.image_channels)

        # encoder networks
        self.image_encoder = ImageEncoder(image_size=self.image_size,
                                          image_channels=self.image_channels,
                                          cnn_hidden_size=self.cnn_hidden_size,
                                          d_model=self.d_model)

        # decoder networks
        self.image_decoder = ImageDecoder(image_size=self.image_size,
                                          vocab_size=self.vocab_size,
                                          slot_size=self.slot_size,
                                          d_model=self.d_model,
                                          num_decoder_heads=self.num_decoder_heads,
                                          num_decoder_layers=self.num_decoder_layers,
                                          num_blocks=self.num_blocks,
                                          dropout=self.dropout)

        # sysbinder
        self.sysbinder = SysBinder(num_iterations=self.num_iterations, 
                                   num_slots=self.num_slots,
                                   input_size=self.d_model, 
                                   slot_size=self.slot_size, 
                                   mlp_hidden_size=self.mlp_hidden_size, 
                                   num_prototypes=self.num_prototypes, 
                                   num_blocks=self.num_blocks)

        # cos anneal
        self.global_step = 0
        self.tau_start = tau_start
        self.tau_final = tau_final
        self.tau_steps = tau_steps

    def forward(self, image, train=False):
        """
        image: B, C, H, W
        tau: float
        """
        outputs = dict()

        B, C, H, W = image.size()

        tau = cosine_anneal(self.global_step,
                            self.tau_start,
                            self.tau_final,
                            0,
                            self.tau_steps)

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(image), dim=1)  # B, vocab_size, H_enc, W_enc
        z_soft = gumbel_softmax(z_logits, tau, False, dim=1)  # B, vocab_size, H_enc, W_enc
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()  # B, vocab_size, H_enc, W_enc
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H_enc * W_enc, vocab_size
        z_emb = self.image_decoder.dict(z_hard)  # B, H_enc * W_enc, d_model
        z_emb = torch.cat([self.image_decoder.bos.expand(B, -1, -1), z_emb], dim=1)  # B, 1 + H_enc * W_enc, d_model
        z_emb = self.image_decoder.decoder_pos(z_emb)  # B, 1 + H_enc * W_enc, d_model

        # dvae recon
        dvae_recon = self.dvae.decoder(z_soft).reshape(B, C, H, W)  # B, C, H, W
        dvae_mse = ((image - dvae_recon) ** 2).sum() / B  # 1

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns = self.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                              # attns: B, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W

        normed_attns = attns / torch.sum(attns, dim=-2, keepdim=True)  # Weighted mean
        masks = torch.mean(attns, dim=2).reshape(B, self.num_slots, H, W)[..., None] # B, num_slots, H, W, 1
        
        if not train:
            recon_transformer = self.decode(slots)
            recon_transformer = recon_transformer.reshape(B, C, H, W)
            outputs["recon_combined"] = recon_transformer
            outputs["recons"] = recon_transformer.permute(0, 2, 3, 1)[:, None, ...] * masks # B, num_slots, H, W, C

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, self.num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, self.num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # decode
        pred = self.image_decoder.tf(z_emb[:, :-1], slots)   # B, H_enc * W_enc, d_model
        pred = self.image_decoder.head(pred)  # B, H_enc * W_enc, vocab_size
        cross_entropy = -(z_hard * torch.log_softmax(pred, dim=-1)).sum() / B  # 1

        outputs["dvae_recon"] = dvae_recon.clamp(0., 1.) # B, C, H, W
        outputs["cross_entropy"] = cross_entropy 
        outputs["dvae_mse"] = dvae_mse
        outputs["attns"] = attns.reshape(attns.shape[0], attns.shape[1], -1) # B, num_slots, C(==1), N_in(==H*W)
        outputs["normed_attns"] = normed_attns.reshape(attns.shape[0], attns.shape[1], attns.shape[2], -1) # B, num_slots, C(==1), N_in(==H*W)
        outputs["masks"] = masks # B, num_slots, H, W, 1

        if train:
            self.global_step += 1
        
        return outputs


    def encode(self, image):
        """
        image: B, C, H, W
        """
        B, C, H, W = image.size()

        # sysbinder
        emb = self.image_encoder.cnn(image)  # B, cnn_hidden_size, H, W
        emb = self.image_encoder.pos(emb)  # B, cnn_hidden_size, H, W
        H_enc, W_enc = emb.shape[-2:]

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, cnn_hidden_size
        emb_set = self.image_encoder.mlp(self.image_encoder.layer_norm(emb_set))  # B, H * W, cnn_hidden_size
        emb_set = emb_set.reshape(B, H_enc * W_enc, self.d_model)  # B, H * W, cnn_hidden_size

        slots, attns = self.sysbinder(emb_set)  # slots: B, num_slots, slot_size
                                                              # attns: B, num_slots, num_inputs

        attns = attns\
            .transpose(-1, -2)\
            .reshape(B, self.num_slots, 1, H_enc, W_enc)\
            .repeat_interleave(H // H_enc, dim=-2)\
            .repeat_interleave(W // W_enc, dim=-1)  # B, num_slots, 1, H, W
        attns_vis = image.unsqueeze(1) * attns + (1. - attns)  # B, num_slots, C, H, W
        
        return slots, attns_vis, attns

    def decode(self, slots):
        """
        slots: B, N, slot_size
        """
        B, num_slots, slot_size = slots.size()
        H_enc, W_enc = (self.image_size // 4), (self.image_size // 4)
        gen_len = H_enc * W_enc

        # block coupling
        slots = self.image_decoder.slot_proj(slots)  # B, num_slots, num_blocks * d_model
        slots = slots + self.image_decoder.block_pos_proj(self.image_decoder.block_pos)  # B, num_slots, num_blocks * d_model
        slots = slots.reshape(B, num_slots, self.num_blocks, -1)  # B, num_slots, num_blocks, d_model
        slots = self.image_decoder.block_coupler(slots.flatten(end_dim=1))  # B * num_slots, num_blocks, d_model
        slots = slots.reshape(B, num_slots * self.num_blocks, -1)  # B, num_slots * num_blocks, d_model

        # generate image tokens auto-regressively
        z_gen = slots.new_zeros(0)
        input = self.image_decoder.bos.expand(B, 1, -1)
        for t in range(gen_len):
            decoder_output = self.image_decoder.tf(
                self.image_decoder.decoder_pos(input),
                slots
            )
            z_next = F.one_hot(self.image_decoder.head(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            input = torch.cat((input, self.image_decoder.dict(z_next)), dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, H_enc, W_enc)
        gen_transformer = self.dvae.decoder(z_gen)

        return gen_transformer.clamp(0., 1.)

    def reconstruct_autoregressive(self, image):
        """
        image: batch_size x image_channels x H x W
        """
        B, C, H, W = image.size()
        slots, attns, _ = self.encode(image)
        recon_transformer = self.decode(slots)
        recon_transformer = recon_transformer.reshape(B, C, H, W)

        return recon_transformer



    

if __name__ == "__main__":
    _ = SysBinderAutoEncoder()

