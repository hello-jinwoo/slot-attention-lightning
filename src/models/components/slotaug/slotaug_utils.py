from typing import Any, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_inv_insts(insts):
    """
    Args:
        `insts`: (B, K, 11)
    """
    new_insts = insts.clone()
    new_insts[:, :, 2:4] = -insts[:, :, 2:4]  # translate
    new_insts[:, :, 4:5] = 1/insts[:, :, 4:5] # scale
    new_insts[:, :, 5:6] = -insts[:, :, 5:6]  # hue
    new_insts[:, :, 6:8] = 1/insts[:, :, 6:8]   # saturation and lightness
    return new_insts

def adjust_insts(insts, attns, normed_attns):
    """
    Args:
        `insts`: (B, K, 11)
        `attn`: (B, K, N_heads, N_in)
    """
    origin = torch.tensor([0.5, 0.5])[None, None].to(attns.device)
    obj_pos = get_obj_pos_from_attns(normed_attns) # (B, K, 2)
    
    # TODO: do we need masking out no_obj?
    # TODO: search for the threshold values, how to make them configurable
    # TODO: should we deal with background?
    obj_pos, obj_exist_mask = filter_no_obj(obj_pos=obj_pos, 
                                            attns=attns, 
                                            no_obj_thrs=0.01,  
                                            replace_value=0.5)
    # adjust obj_pos
    # insts[:, :, :2] = obj_pos.clone() # unnecessary
    # adjust translate according to the obj_pos and scale
    adjusted_insts = insts.detach().clone()

    # TODO: we now assume there is no inst that have simultaneous `scale` and `flip`
    #       we may have to deal with it if needed
    if torch.sum(insts[:, :, 4:5] != 1) > 0: # scale
        # adjust (x,y) according to the vector of (obj_pos - origin)
        adjusted_insts[:, :, 2:4] = insts[:, :, 2:4] + ((0.5 + (obj_pos - 0.5) * insts[:, :, 4:5]) - obj_pos)

    # if torch.sum(insts[:, :, 9:10] > 0) > 0: # hflip
    #     # adjust x 
    #     adjusted_insts[:, :, 3:4] = 1 - obj_pos[:, :, 0:1] * 2
    # if torch.sum(insts[:, :, 10:11] > 0) > 0: # vflip
    #     # adjust y
    #     adjusted_insts[:, :, 4:5] = 1 - obj_pos[:, :, 1:2] * 2
    # if torch.sum(insts[:, :, 2:3] > 0) > 0: # rotation
    #     # convert rotation angle to radian
    #     t = adjusted_insts[:, :, 2:3] * 2 * math.pi
    #     # rotation matrix R
    #     R = torch.stack([
    #         torch.stack([torch.cos(t), -torch.sin(t)], dim=-1),
    #         torch.stack([torch.sin(t), torch.cos(t)], dim=-1),
    #     ], dim=-2)
    #     # normalize obj pos relative to origin
    #     normed_obj_pos = obj_pos - origin 
    #     # rotate normed obj pos with R
    #     normed_obj_pos_rotated = torch.einsum('bijk,bikl->bijl',[normed_obj_pos[:,:,None], R.squeeze(2)]).squeeze(2)
    #     # add origin back
    #     obj_pos_rotated = normed_obj_pos_rotated + origin

    #     adjusted_insts[:, :, 3:5] = obj_pos_rotated - obj_pos

    return adjusted_insts

def masks_to_attns(masks):
    attns = torch.einsum("bkhwi->bhwik", masks)
    attns = attns.reshape(attns.shape[0], -1, attns.shape[-1]).unsqueeze(1) # (B, N_head=1, N_in, K)
    normed_attns = attns / torch.sum(attns, dim=-2, keepdim=True) # (B, N_head=1, N_in, K)
    attns = torch.einsum("bhik->bkhi", attns) # (B, K, N_head, N_in)
    normed_attns = torch.einsum("bhik->bkhi", normed_attns) # (B, K, N_head, N_in)
    return attns, normed_attns

def filter_no_obj(obj_pos, attns, no_obj_thrs=0.01, replace_value=-10):
    attns = torch.mean(attns, dim=2) # (B, K, N_head, N_in) -> (B, K, N_in)
    # TODO: better masking out strategy?
    obj_exist_mask = (torch.mean((torch.eq(torch.max(attns, dim=1, keepdim=True).values, attns).float()), dim=2) > no_obj_thrs).float() # (B, K)
    filtered_obj_pos = torch.full_like(obj_pos, replace_value).to(attns.device) * (1 - obj_exist_mask[..., None]) + \
                       obj_pos * obj_exist_mask[..., None]# (B, K, 2), obj_pos default 0.5
    
    # # if you want to check (print) the filtering results
    # print("obj_pos", obj_pos)
    # print("obj_exist_mask", obj_exist_mask)
    # print("filtered_obj_pos", filtered_obj_pos)
    
    return filtered_obj_pos, obj_exist_mask

def filter_background(
        obj_pos: torch.tensor = torch.tensor([]),
        attns: torch.tensor = torch.tensor([]),
        replace_value: float = 0.5, 
    ):
        attns_sum = torch.sum(torch.mean(attns, dim=-2), dim=-1) # (1, N_slots) avg over N_heads and then sum over N_in
        background_mask = (attns_sum == torch.max(attns_sum, dim=-1, keepdim=True).values).float()  # (1, N_slots)
        obj_pos_filtered = torch.full_like(obj_pos, replace_value).to(attns.device) * background_mask[..., None] + \
                           obj_pos * (1 - background_mask[..., None]) # (1, N_slots, 2)
        return obj_pos_filtered, background_mask

def get_coord_map(w, h):
    x = torch.arange(w) / w
    x = torch.stack([x] * h)
    y = torch.arange(h) / h
    y = torch.stack([y] * w)
    return torch.stack([x, y.T], dim=-1) # (h, w, 2), 2 -> (x,y)

def get_obj_pos_from_attns(normed_attns):
    normed_attns = torch.mean(normed_attns, dim=2) # (B, K, N_head, N_in) -> (B, K, N_in)
    B, K, N_in = normed_attns.shape # here B will be 1

    # make coord map
    # TODO: to all image size
    w = int(N_in ** 0.5)
    h = int(N_in ** 0.5)
    coord_map = get_coord_map(w, h).to(normed_attns.device) # (h, w, 2)
    coord_map = coord_map.reshape(1, 1, -1, 2) # (1, 1, N_in, 2)

    return torch.sum(coord_map * normed_attns.unsqueeze(-1), dim=2) # (B, K, N_in, 2) -> (B, K, 2)

def rearrange_slots(slots_src, obj_pos_src, obj_pos_tgt):
    matched_idxs = matching_pairs(obj_pos_src, obj_pos_tgt) # (B, K)
    slots_rearranged = rearrange_by_matched_idx(slots_src, matched_idxs)
    return slots_rearranged # (B, K, D_slot)
    
def rearrange_insts(insts, obj_pos):
    obj_pos_insts = insts[:, :, 0:2]
    matched_idxs = matching_pairs(obj_pos_insts, obj_pos) # (B, K)
    rearranged_insts = rearrange_by_matched_idx(insts, matched_idxs)
    return rearranged_insts

def matching_pairs(src, tgt):
    return matching_pairs_(tgt, src)

def matching_pairs_(tgt, src):
    # src, tgt (B, K, 2), here B will be 1
    cost_map = torch.cdist(tgt, src).cpu().detach().numpy() # (B, K, K)
    matched_idxs = np.array([linear_sum_assignment(cost_map[i])[1] for i in range(len(tgt))]) # [B, K]
    return matched_idxs

def rearrange_by_matched_idx(src, matched_idxs):
    src = src.clone()
    batch_indices = [i // src.shape[1] for i in range(src.shape[0] * src.shape[1])]
    return src[range(len(src))][batch_indices, matched_idxs.reshape(-1)].reshape(src.shape)

def get_slot_dist(
    slots_src: torch.tensor = torch.tensor([]),
    obj_pos_src: torch.tensor = torch.tensor([]),
    slots_tgt: torch.tensor = torch.tensor([]),
    obj_pos_tgt: torch.tensor = torch.tensor([]),
    metric: str = "rmse", 
    reduction: bool = False, 
):
    """
    Parameters:
        slots_src: (B, K, D_slot)
        slots2: (B, K, D_slot)
    """
    slots_src = rearrange_slots(slots_src, obj_pos_src, obj_pos_tgt)
    if metric == "rmse": 
        dists = torch.sqrt(torch.sum(F.mse_loss(slots_src, slots_tgt, reduction="none"), dim=-1))
    elif metric == "mse": 
        dists = torch.sum(F.mse_loss(slots_src, slots_tgt, reduction="none"), dim=-1)
    elif metric == "cosine":
        dists = F.cosine_similarity(slots_src, slots_tgt, dim=2)
    
    if reduction:
            return torch.mean(dists)
    
    return dists 
    
# class TranslationEncoder(nn.Module):
class PositionEncoder(nn.Module):
    def __init__(self, slot_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.mlp(x)

class ScaleEncoder(nn.Module):
    def __init__(self, slot_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.mlp(x)
    
class ColorEncoder(nn.Module):
    def __init__(self, slot_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.mlp(x)
    
class RotationEncoder(nn.Module):
    def __init__(self, slot_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(1, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.mlp(x)
    
class FlipEncoder(nn.Module):
    def __init__(self, slot_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.mlp(x)

# Kernel Normalized Kernel
class WNConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, norm_type="softmax", use_normed_logits=False):
        super(WNConv, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.norm_type = norm_type

    def forward(self, input):
        o_c, i_c, k1, k2 = self.weight.shape
        weight = torch.reshape(self.weight.data, (o_c, i_c, k1 * k2))
        weight = weight / torch.linalg.norm(weight, dim=-1, keepdim=True) # norm logits to 1

        if 'linear' in self.norm_type:
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        elif 'softmax' in self.norm_type:
            weight = F.softmax(weight, dim=-1)

        self.weight = nn.Parameter(torch.reshape(weight, (o_c, i_c, k1, k2)))

        # we don't recommend ver lower than 1.7
        if '1.7' in torch.__version__:
            return self._conv_forward(input, self.weight)
        else:
            return self._conv_forward(input, self.weight, self.bias)