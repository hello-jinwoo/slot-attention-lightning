import numpy as np
import torch
from torch import nn

from src.models.components.sysbinder.sysbinder_utils import *


class SysBinder(nn.Module):
    
    def __init__(self, 
                 num_iterations, 
                 num_slots,
                 input_size, 
                 slot_size, 
                 mlp_hidden_size, 
                 num_prototypes, 
                 num_blocks,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_blocks = num_blocks
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = BlockLayerNorm(slot_size, num_blocks)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = BlockGRU(slot_size, slot_size, num_blocks)
        self.mlp = nn.Sequential(
            BlockLinear(slot_size, mlp_hidden_size, num_blocks),
            nn.ReLU(),
            BlockLinear(mlp_hidden_size, slot_size, num_blocks))
        self.prototype_memory = BlockPrototypeMemory(num_prototypes, num_blocks, slot_size)

    def forward(self, inputs):
        B, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.view(-1, self.slot_size),
                slots_prev.view(-1, self.slot_size)
            )
            slots = slots.view(-1, self.num_slots, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
            slots = self.prototype_memory(slots)

        return slots, attn_vis