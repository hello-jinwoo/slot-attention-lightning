from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

import timm

from einops import rearrange

class SoftPositionEmbed(nn.Module):
    """Builds the soft position embedding layer with learnable projection.

    Args:
        hid_dim (int): Size of input feature dimension.
        resolution (tuple): Tuple of integers specifying width and height of grid.
    """

    def __init__(
        self,
        hid_dim: int = 64,
        resolution: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        self.grid = self.build_grid(resolution)

    def forward(self, inputs):
        self.grid = self.grid.to(inputs.device)
        grid = self.embedding(self.grid).to(inputs.device)
        return inputs + grid

    def build_grid(self, resolution):
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        hid_dim: int = 64,
        enc_depth: int = 4,
        use_pe: bool = True,
    ):
        super().__init__()
        assert enc_depth > 2, "Depth must be larger than 2."

        convs = nn.ModuleList([nn.Conv2d(3, hid_dim, 5, padding="same"), nn.ReLU()])
        for _ in range(enc_depth - 2):
            convs.extend([nn.Conv2d(hid_dim, hid_dim, 5, padding="same"), nn.ReLU()])
        convs.append(nn.Conv2d(hid_dim, hid_dim, 5, padding="same"))
        self.convs = nn.Sequential(*convs)

        self.use_pe = use_pe
        if use_pe:
            self.encoder_pos = SoftPositionEmbed(hid_dim, (img_size, img_size))
        self.layer_norm = nn.LayerNorm([img_size * img_size, hid_dim])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x):
        x = self.convs(x)  # [B, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W ,D]
        if self.use_pe:
            x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

def convert_batchnorm_to_groupnorm(module, num_groups=32):
    """
    Recursively convert all BatchNorm modules to GroupNorm modules in a model.
    """
    if isinstance(module, torch.nn.BatchNorm2d):
        num_channels = module.num_features
        return torch.nn.GroupNorm(num_groups, num_channels)

    for name, child in module.named_children():
        module.add_module(name, convert_batchnorm_to_groupnorm(child, num_groups))

    return module

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        feat_size: int = 16,
        hid_dim: int = 512,
        enc_depth: int = 4,
        use_pe: bool = True,
        norm_type: str = "bn"
    ):
        super().__init__()
        assert enc_depth in [34, 50], "Depth must be either 34 or 50."

        if enc_depth == 34:
            resnet34 = timm.create_model('resnet34', features_only=True, pretrained=False)
            if norm_type == "gn":
                resnet34 = convert_batchnorm_to_groupnorm(resnet34)
            self.resnet = ModifiedResNet(resnet34)
            hid_dim = 512
        elif enc_depth == 50:
            resnet50 = timm.create_model('resnet50', features_only=True, pretrained=False)
            if norm_type == "gn":
                resnet50 = convert_batchnorm_to_groupnorm(resnet50)
            self.resnet = ModifiedResNet(resnet50)
            hid_dim = 2048
        
        self.use_pe = use_pe
        if use_pe:
            self.encoder_pos = SoftPositionEmbed(hid_dim, (feat_size, feat_size))

        self.layer_norm = nn.LayerNorm([feat_size * feat_size, hid_dim])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x):
        x = self.resnet(x)  # [B, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W ,D]
        if self.use_pe:
            x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class ModifiedResNet(torch.nn.Module):
    def __init__(self, model):
        super(ModifiedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same', bias=False)
        #self.bn1 = model.bn1
        self.gn1 = torch.nn.GroupNorm(32, 64, eps=1e-6)
        self.relu = model.act1
        #self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        slot_dim: int = 64,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.dec_init_size = dec_init_size
        self.decoder_pos = SoftPositionEmbed(slot_dim, (dec_init_size, dec_init_size))

        D_slot = slot_dim
        D_hid = dec_hid_dim
        upsample_step = int(np.log2(img_size // dec_init_size))

        deconvs = nn.ModuleList()
        count_layer = 0
        for _ in range(upsample_step):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot,
                        D_hid,
                        5,
                        stride=(2, 2),
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        for _ in range(dec_depth - upsample_step - 1):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot, D_hid, 5, stride=(1, 1), padding=2
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(D_hid, 4, 3, stride=(1, 1), padding=1))
        # deconvs.append(nn.ConvTranspose2d(D_hid, 4, 1, stride=(1, 1))) # isa dec
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.dec_init_size, self.dec_init_size, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconvs(x)
        x = x[:, :, : self.img_size, : self.img_size]
        x = x.permute(0, 2, 3, 1)
        return x

# For SRTDecoder (including Transformer)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x) # (B, H*W, dim) -> (B, H*W, z_dim)
            #q = q[None], q[: None]
            k, v = self.to_kv(z).chunk(2, dim=-1) # (B, K, dim) -> (B, K, z_dim), (B, K, z_dim)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # dots: (B, heads, H*W, K)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, kv_dim=kv_dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        # x : queries, z : slots (keys and values)
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x
    
class SRTDecoderContent(nn.Module):
    def __init__(self, num_att_blocks=2, out_dims=3,
                 z_dim=768, input_mlp=False, output_mlp=True, img_size=128, slot_dim=64):
        super().__init__()

        self.img_size = img_size

        self.input_mlp_slot = nn.Sequential(
            nn.Linear(slot_dim, 360),
            nn.ReLU(),
            nn.Linear(360, 180))
        
        self.input_mlp_query = nn.Sequential(
            nn.Linear(2, 180),
            nn.ReLU(),
            nn.Linear(180, 180))

        self.transformer = Transformer(180, depth=num_att_blocks, heads=12, dim_head=z_dim // 12,
                                       mlp_dim=z_dim * 2, selfatt=False, kv_dim=180)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(180, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

        x_coords = torch.arange(self.img_size)
        y_coords = torch.arange(self.img_size)
        x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords)
        self.coords_tensor = torch.stack((x_mesh.flatten(), y_mesh.flatten()), axis=1).to(dtype=torch.float32)
        self.coords_tensor.requires_grad_(False)

    def forward(self, slots):
        """
        Args:
            z: scene encoding [batch_size, num_patches, patch_dim]
            x: query camera positions [batch_size, num_rays, 3]
            rays: query ray directions [batch_size, num_rays, 3]
        """
        self.coords_tensor = self.coords_tensor.to(slots.device) # (H*W, 2)

        queries = self.input_mlp_query(self.coords_tensor) # (H*W, 2) -> (H*W, 180)
        queries = queries.expand(slots.shape[0], queries.shape[0], queries.shape[1]) # (H*W, 180) -> (B, H*W, 180)

        slots = self.input_mlp_slot(slots) # (B, K, D_slot) -> (B, K, 180)

        # if self.input_mlp is not None:
        #     queries = self.input_mlp(queries)

        output = self.transformer(queries, slots)
        if self.output_mlp is not None:
            output = self.output_mlp(output)

        return output
    
class SRTDecoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        slot_dim: int = 64,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
    ):
        super().__init__()

        # 기본 설정 값 이용
        # num_att_blocks = depth of transformer decoder
        self.num_att_blocks = 2
        self.img_size = img_size

        # positional encoding 부분 삭제
        # out_dims : 마지막 mlp 태우고 나오는 차원 값
        # z_dim : feed forward network의 hidden state dimension

        self.model = SRTDecoderContent(num_att_blocks=self.num_att_blocks,
                                       out_dims=3, z_dim=768, slot_dim=slot_dim,
                                       input_mlp=True, output_mlp=True, img_size=self.img_size)
        

    def forward(self, slots):
        # x : (B, K, D_slot) ->  (3, height, width).
        output_img = self.model(slots)

        return output_img

class SRTDecoder_Slotwise(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        ray_size: int = 128,
        slot_dim: int = 64,
        query_dim: int = 64,
        dec_hid_dim: int = 128,
        output_mlp: bool = True,
        out_dims: int = 4,
        num_split: int = 4,
    ):
        super().__init__()

        # 기본 설정 값 이용
        # num_att_blocks = depth of transformer decoder
        self.num_att_blocks = 2
        self.img_size = img_size
        self.ray_size = ray_size
        self.query_dim = query_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_split = num_split
        self.input_mlp_slot = nn.Linear(slot_dim, self.query_dim * self.num_split) # slot will be split into num_split
        
        self.input_mlp_query = nn.Linear(2, self.query_dim)

        self.transformer = Transformer(self.query_dim, depth=self.num_att_blocks, heads=8, dim_head=self.dec_hid_dim // 8,
                                       mlp_dim=self.dec_hid_dim * 2, selfatt=False, kv_dim=self.query_dim)

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.query_dim, self.query_dim // 2),
                nn.ReLU(),
                nn.Linear(self.query_dim // 2 , out_dims))
        else:
            self.output_mlp = None

        x_coords = torch.arange(self.ray_size)
        y_coords = torch.arange(self.ray_size)
        x_mesh, y_mesh = torch.meshgrid(x_coords, y_coords)
        self.coords_tensor = torch.stack((x_mesh.flatten(), y_mesh.flatten()), axis=1).to(dtype=torch.float32)
        self.coords_tensor.requires_grad_(False)
        # positional encoding 부분 삭제
        # out_dims : 마지막 mlp 태우고 나오는 차원 값
        # z_dim : feed forward network의 hidden state dimension


    def forward(self, slots):
        B, K, D_slot = slots.shape
        self.coords_tensor = self.coords_tensor.to(slots.device) # (H*W, 2)

        queries = self.input_mlp_query(self.coords_tensor) # (H*W, 2) -> (H*W, query_dim)
        queries = queries.expand(B*K, queries.shape[0], queries.shape[1]) # (H*W, query_dim) -> (B*K, H*W, query_dim)

        slots = self.input_mlp_slot(slots) # (B, K, D_slot) -> (B, K, query_dim * num_split)
        ind_slots = slots.reshape(B*K, self.num_split, self.query_dim) # (B*K, num_split, query_dim)

        output = self.transformer(queries, ind_slots)

        if self.output_mlp is not None:
            output = self.output_mlp(output)
            output = output.reshape(B*K, self.ray_size, self.ray_size, -1) # (B*K, H, W, 4)

        return output
