import torch
from torch import nn
from src.models.components.slotaug.slotaug_utils import WNConv

class SlotAttention(nn.Module):
    """Slot Attention module.

    Args:
        num_slots: int - Number of slots in Slot Attention.
        iterations: int - Number of iterations in Slot Attention.
        num_attn_heads: int - Number of multi-head attention in Slot Attention,
    """

    def __init__(
        self,
        num_slots: int = 7,
        num_iterations: int = 3,
        num_attn_heads: int = 1,
        slot_dim: int = 64,
        hid_dim: int = 64,
        mlp_hid_dim: int = 128,
        ark_size: int = 5,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.iterations = num_iterations
        self.num_attn_heads = num_attn_heads
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.eps = eps

        self.scale = (num_slots // num_attn_heads) ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, self.slot_dim))

        self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_slot = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hid_dim, self.slot_dim),
        )

        self.ark_size = ark_size 
        if ark_size > 0: 
            self.ark = WNConv(
                in_channels=1,
                out_channels=1,
                kernel_size=ark_size,
                padding=ark_size // 2,
                bias=False
            )

    def forward(self, inputs):
        outputs = dict()

        B, N_in, D_in = inputs.shape
        K = self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_attn_heads

        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_sigma.expand(B, K, -1)
        slots = torch.normal(mu, torch.abs(sigma) + self.eps)

        inputs = self.norm_input(inputs)

        # `k`, `v`: (B, N_heads, N_in, D_slot // N_heads).
        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)

        attns = list()
        for iter_idx in range(self.iterations):
            slots_prev = slots
            slots = self.norm_slot(slots)

            q = self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
            # `q`: (B, N_heads, K, slot_D // N_heads)

            attns_logits = torch.einsum("bhid, bhjd->bhij", k, q) * self.scale

            # ARK
            if self.ark_size > 0:
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

        outputs["slots"] = slots
        outputs["attns"] = attns
        outputs["normed_attns"] = normed_attns
        # `attns`: (B, K, N_heads, N_in)
        # `normed_attns`: (B, K, N_heads, N_in)

        return outputs


if __name__ == "__main__":
    _ = SlotAttention()
