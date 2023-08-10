import torch
from torch import nn

from src.models.components.slotaug.slotaug_utils import *

torch.autograd.set_detect_anomaly(True)
class SlotAug(nn.Module):
    """Slot Attention module.

    Args:
        num_slots: int - Number of slots in Slot Attention.
        num_iterations: int - Number of iterations in Slot Attention.
        num_attn_heads: int - Number of multi-head attention in Slot Attention,
    """

    def __init__(
        self,
        num_slots: int = 7,
        num_iter: int = 3,
        num_iter_insts: int = 1,
        num_attn_heads: int = 1,
        slot_dim: int = 64,
        hid_dim: int = 64,
        mlp_hid_dim: int = 128,
        ark_size: int = 5,
        aux_identity: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.num_iter = num_iter
        self.num_iter_insts = num_iter_insts
        self.num_attn_heads = num_attn_heads
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.aux_identity = aux_identity
        self.eps = eps

        self.insts_dim = 8

        self.scale = (num_slots // num_attn_heads) ** -0.5

        self.slots_mu = nn.Parameter(torch.rand(1, 1, self.slot_dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, self.slot_dim))

        self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_slot = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)
        self.norm_inst_list = nn.ModuleList([
            nn.LayerNorm(self.slot_dim), # rotate
            nn.LayerNorm(self.slot_dim), # translate
            nn.LayerNorm(self.slot_dim), # scale
            nn.LayerNorm(self.slot_dim), # color
            nn.LayerNorm(self.slot_dim), # flip
        ])
        
        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hid_dim, self.slot_dim),
        )
        
        self.ark = WNConv(
            in_channels=1,
            out_channels=1,
            kernel_size=ark_size,
            padding=ark_size // 2,
            bias=False
        )

        self.inst_encoder_list = nn.ModuleList([
            PositionEncoder(slot_dim),
            ScaleEncoder(slot_dim),
            ColorEncoder(slot_dim),
        ])

        self.inst_contents = ['translate', 'scale', 'color'] # TODO: split flip into hflip and vflip?
        self.inst_content_idxs=[[2,4], [4,5], [5,8]]

        self.aux_identity_insts = torch.zeros(1, num_slots, self.insts_dim, dtype=torch.float32)
        self.aux_identity_insts[:, :, 4:5] = 1 # scale
        self.aux_identity_insts[:, :, 6:8] = 1 # saturation and light

    def apply_insts(self, insts, slots):
        """
        Args:
            `insts`: (B, K, insts_dim)
            `slots`: (B, K, D_slot)
        """
        for content_idx in range(len(self.inst_contents)):
            inst_content_idx = self.inst_content_idxs[content_idx]
            inst = insts[:, :, inst_content_idx[0]:inst_content_idx[1]]
            inst_slots = self.inst_encoder_list[content_idx](inst)
            
            slots = slots + self.norm_inst_list[content_idx](inst_slots)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots

    def update_slots(self, k, v, slots): 
        B, N_heads, N_in, _ = k.shape
        _, K, D_slot = slots.shape

        # JW question: why is the slots_prev not set to be normed version of slots?
        slots_prev = slots
        slots = self.norm_slot(slots)

        q = self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
        # `q`: (B, N_heads, K, slot_D // N_heads)

        attns_logits = torch.einsum("bhid, bhjd->bhij", k, q) * self.scale

        # ARK
        attns_logits = attns_logits.permute(0, 3, 1, 2) # [B, K, N_heads, N_in]
        attns_logits = attns_logits.reshape(-1, N_in)[:, None, :] # [B*K*N_head, 1, N_in]
        img_size = int(N_in ** 0.5)
        attns_logits = attns_logits.reshape(-1, 1, img_size, img_size) # [B*K*N_heads, 1, img_size, img_size]
        attns_logits = self.ark(attns_logits) # [B*K*N_heads, 1, img_size, img_size]
        attns_logits = attns_logits.reshape(B, K, N_heads, N_in) # [B, K, N_heads, N_in]
        attns_logits = attns_logits.permute(0, 2, 3, 1) # (B, N_heads, N_in, K)

        attns = attns_logits.softmax(dim=-1) + self.eps  # Normalization over slots
        # `attn`: (B, N_heads, N_in, K)

        normed_attns = attns / torch.sum(attns, dim=-2, keepdim=True)  # Weighted mean
        # `normed_attns`: (B, N_heads, N_in, K)

        updates = torch.einsum("bhij,bhid->bhjd", normed_attns, v)
        # `updates`: (B, N_heads, K, slot_D // N_heads)
        updates = updates.transpose(1, 2).reshape(B, K, -1)
        # `updates`: (B, K, slot_D)

        slots = self.gru(updates.reshape(-1, D_slot), slots_prev.reshape(-1, D_slot))
        slots = slots.reshape(B, -1, D_slot)
        slots = slots + self.mlp(self.norm_mlp(slots))

        attns = torch.einsum("bhik->bkhi", attns)
        normed_attns = torch.einsum("bhik->bkhi", normed_attns)
        return slots, attns, normed_attns
        # `attns`: (B, K, N_heads, N_in)
        # `normed_attns`: (B, K, N_heads, N_in)

    def spatial_binding(self, inputs):
        B, N_in, _ = inputs.shape
        K = self.num_slots
        N_heads = self.num_attn_heads

        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_sigma.expand(B, K, -1)
        slots = torch.normal(mu, torch.abs(sigma) + self.eps)

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        # `k`, `v`: (B, N_heads, N_in, D_slot // N_heads)

        for iter_idx in range(self.num_iter):
            slots, attns, normed_attns = self.update_slots(
                k=k, 
                v=v,
                slots=slots,
            )

        outputs = {
            "slots": slots, 
            # `slots`: (B, K, D_slot)
            "attns": attns, 
            # `attns`: (B, K, N_heads, N_in), after softmax before normalization
            "normed_attns": normed_attns,  
            # `normed_attns`: (B, K, N_heads, N_in), after softmax and normalization
        }

        return outputs

    # forward() only used for training 
    def forward(self, inputs_ori, inputs_aug, insts_ori2aug, insts_aug2ori):     
        B = inputs_ori.shape[0]

        outputs = self.spatial_binding(inputs=inputs_ori)
        slots_ori = outputs["slots"]
        if inputs_aug != None:
            outputs_aug = self.spatial_binding(inputs=inputs_aug)
            slots_aug = outputs_aug["slots"]
        
        if self.aux_identity:
            slots_ori = self.apply_insts(
                insts=self.aux_identity_insts.repeat(B, 1, 1).to(insts_ori2aug.device), 
                slots=slots_ori,
            )
        
        insts_ori2aug = adjust_insts(
            insts=insts_ori2aug, 
            attns=outputs["attns"], 
            normed_attns=outputs["normed_attns"]
        )

        slots_ori2aug = self.apply_insts(
            insts=insts_ori2aug, 
            slots=slots_ori, 
        )

        slots_ori_revisited = self.apply_insts(
            insts=get_inv_insts(insts_ori2aug),
            slots=slots_ori2aug,
        )
        outputs["slots_ori"] = slots_ori
        outputs["slots_ori2aug"] = slots_ori2aug
        outputs["slots_ori_revisited"] = slots_ori_revisited

        outputs["attns_ori"] = outputs["attns"]
        outputs["normed_attns_ori"] = outputs["normed_attns"]
        if inputs_aug != None:
            outputs["slots_aug"] = slots_aug
            outputs["attns_aug"] = outputs_aug["attns"]
            outputs["normed_attns_aug"] = outputs_aug["normed_attns"]
        else:
            outputs["slots_aug"] = slots_ori
            outputs["attns_aug"] = outputs["attns"]
            outputs["normed_attns_aug"] = outputs["normed_attns"]

        return outputs

if __name__ == "__main__":
    _ = SlotAug()