import torch
import torch.nn as nn
import numpy as np
import time
import einops
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from tools.utils import vision_features, save_image_heat_map, save_image_heat_map_list
from typing import Union, Sequence
from network.wtconv2d import WTConv2d

class Padding_tensor(nn.Module):
    def __init__(self, patch_size):
        super(Padding_tensor, self).__init__()
        self.patch_size = patch_size
    def forward(self, x):
        b, c, h, w = x.shape
        h_patches = int(np.ceil(h / self.patch_size))
        w_patches = int(np.ceil(w / self.patch_size))
        h_padding = np.abs(h - h_patches * self.patch_size)
        w_padding = np.abs(w - w_patches * self.patch_size)
        reflection_padding = [0, w_padding, 0, h_padding]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x = reflection_pad(x)
        return x, [h_patches, w_patches, h_padding, w_padding]

class PatchEmbed_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.padding_tensor = Padding_tensor(patch_size)

    def forward(self, x):
        b, c, h, w = x.shape
        x, patches_paddings = self.padding_tensor(x)
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        # -------------------------------------------
        patch_matrix = None
        for i in range(h_patches):
            for j in range(w_patches):
                patch_one = x[:, :, i * self.patch_size: (i + 1) * self.patch_size,
                            j * self.patch_size: (j + 1) * self.patch_size]
                patch_one = patch_one.reshape(-1, c, 1, self.patch_size, self.patch_size)
                if i == 0 and j == 0:
                    patch_matrix = patch_one
                else:
                    patch_matrix = torch.cat((patch_matrix, patch_one), dim=2)
        return patch_matrix, patches_paddings


class Recons_tensor(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    def forward(self, patches_tensor, patches_paddings):
        B, C, N, Ph, Pw = patches_tensor.shape
        h_patches = patches_paddings[0]
        w_patches = patches_paddings[1]
        h_padding = patches_paddings[2]
        w_padding = patches_paddings[3]
        assert N == h_patches * w_patches, \
            f"The number of patches ({N}) doesn't match the Patched_embed operation ({h_patches}*{w_patches})."
        assert Ph == self.patch_size and Pw == self.patch_size, \
            f"The size of patch tensor ({Ph}*{Pw}) doesn't match the patched size ({self.patch_size}*{self.patch_size})."

        patches_tensor = patches_tensor.view(-1, C, N, self.patch_size, self.patch_size)
        # ----------------------------------------
        pic_all = None
        for i in range(h_patches):
            pic_c = None
            for j in range(w_patches):
                if j == 0:
                    pic_c = patches_tensor[:, :, i * w_patches + j, :, :]
                else:
                    pic_c = torch.cat((pic_c, patches_tensor[:, :, i * w_patches + j, :, :]), dim=3)
            if i == 0:
                pic_all = pic_c
            else:
                pic_all = torch.cat((pic_all, pic_c), dim=2)
        b, c, h, w = pic_all.shape
        pic_all = pic_all[:, :, 0:(h - h_padding), 0:(w - w_padding)]
        return pic_all
# -----------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, in_dims=512, attn_p=0., proj_p=0., cross=False):
        super().__init__()
        token_dim = 16
        self.cross = cross
        self.to_query = nn.Linear(dim, token_dim * num_heads)
        self.to_key = nn.Linear(dim, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, dim)

    def forward(self, x):
        if self.cross:
            query = self.to_query(x[0])
            key = self.to_key(x[1])
        else:
            query = self.to_query(x)
            key = self.to_key(x)
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)
        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)
        G = torch.sum(A * query, dim=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])
        out = self.Proj(G * key) + query
        out = self.final(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0., cross=False):
        super().__init__()
        self.cross = cross
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p,
            cross=cross
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
        )

    def forward(self, x):
        if self.cross:
            x_ = [self.norm1(_x) for _x in x]
            out = x[1] + self.attn(x_)
            out = out + self.mlp(self.norm2(out))
            out = [x_[0], out, out]
        else:
            out = x + self.attn(self.norm1(x))
            out = out + self.mlp(self.norm2(out))
        return out
# --------------------------------------------------------------------------------------
class self_atten_module(nn.Module):
    def __init__(self, embed_dim, num_p, depth, num_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0.1, attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p, cross=False)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x_in):
        x = x_in
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x_self = x
        return x_self

class cross_atten_module(nn.Module):
    def __init__(self, embed_dim, num_patches, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, p=0.1, attn_p=0.):
        super().__init__()
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                if i == 0 else
                Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p,
                      cross=True)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x1_ori, x2_ori):
        x1 = x1_ori
        x2 = x2_ori
        x2 = self.pos_drop(x2)
        x = [x1, x2]
        for block in self.blocks:
            x = block(x)
            x[1] = self.norm(x[1])
        x_self = x[1]
        return x_self

class self_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, kernel_size, depth_self, num_heads=16,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)

        self.self_atten1 = self_atten_module(embed_dim, num_patches, depth_self,
                                             num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten2 = self_atten_module(embed_dim, num_patches, depth_self,
                                             num_heads, mlp_ratio, qkv_bias, p, attn_p)


        self.wtblock = WTblock(in_channels=64, out_channels=64, kernel_size=kernel_size)
    def forward(self, x1, x2, last=False):
        x1, x2 = self.wtblock(x1, x2)
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        x_patched2, _ = self.patch_embed_tensor(x2)
        b, c, n, h, w = x_patched1.shape
        x_patched1 = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x_patched2 = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x1_self_patch = self.self_atten1(x_patched1)
        x2_self_patch = self.self_atten2(x_patched2)

        if last is False:
            x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
            x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
            x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W
        else:
            x_self1 = x1_self_patch
            x_self2 = x2_self_patch

        return x_self1, x_self2, patches_paddings

class cross_atten(nn.Module):
    def __init__(self, patch_size, embed_dim, num_patches, depth_self, depth_cross, num_heads,
                 mlp_ratio=4., qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_embed_tensor = PatchEmbed_tensor(patch_size)
        self.recons_tensor = Recons_tensor(patch_size)

        self.cross_atten1 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                               num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.cross_atten2 = cross_atten_module(embed_dim, num_patches, depth_cross,
                                               num_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, patches_paddings):
        x_patched1, patches_paddings = self.patch_embed_tensor(x1)
        x_patched2, _ = self.patch_embed_tensor(x2)

        b, c, n, h, w = x_patched1.shape
        # b, n, c*h*w
        x1_self_patch = x_patched1.transpose(2, 1).contiguous().view(b, n, c * h * w)
        x2_self_patch = x_patched2.transpose(2, 1).contiguous().view(b, n, c * h * w)

        x_in1 = x1_self_patch
        x_in2 = x2_self_patch
        cross1 = self.cross_atten1(x_in1, x_in2)
        cross2 = self.cross_atten2(x_in2, x_in1)
        out = cross1 + cross2
        # reconstruct
        x1_self_patch = x1_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self1 = self.recons_tensor(x1_self_patch, patches_paddings)  # B, C, H, W
        x2_self_patch = x2_self_patch.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        x_self2 = self.recons_tensor(x2_self_patch, patches_paddings)  # B, C, H, W

        cross1 = cross1.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross1_all = self.recons_tensor(cross1, patches_paddings)  # B, C, H, W

        cross2 = cross2.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        cross2_all = self.recons_tensor(cross2, patches_paddings)  # B, C, H, W

        out = out.view(b, n, c, h, w).permute(0, 2, 1, 3, 4)
        out_all = self.recons_tensor(out, patches_paddings)  # B, C, H, W

        return out_all, x_self1, x_self2, cross1_all, cross2_all

class WTblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTblock, self).__init__()
        self.wtconv1 = WTConv2d(in_channels, out_channels, kernel_size, stride, bias, wt_levels, wt_type)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.wtconv2 = WTConv2d(in_channels, out_channels, kernel_size, stride, bias, wt_levels, wt_type)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x1, x2):

        out1 = self.wtconv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu1(out1)

        out2 = self.wtconv2(x2)
        out2 = self.bn2(out2)
        out2 = self.relu2(out2)

        return out1, out2

class cross_encoder(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, num_patches, depth_self, depth_cross, num_heads=8,
                 mlp_ratio=2., qkv_bias=True, p=0.1, attn_p=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.depth_cross = depth_cross
        kernel_size1 = 3
        kernel_size2 = 5
        kernel_size3 = 7
        self.self_atten_block1 = self_atten(self.patch_size, embed_dim, num_patches, kernel_size1, depth_self * 2,
                                            num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten_block2 = self_atten(self.patch_size, embed_dim, num_patches, kernel_size2,depth_self * 2,
                                            num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.self_atten_block3 = self_atten(self.patch_size, embed_dim, num_patches, kernel_size3,depth_self * 2,
                                            num_heads, mlp_ratio, qkv_bias, p, attn_p)

        self.cross_atten_block1 = cross_atten(self.patch_size, embed_dim, self.num_patches, depth_self,
                                              depth_cross, num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.cross_atten_block2 = cross_atten(self.patch_size, embed_dim, self.num_patches, depth_self,
                                              depth_cross, num_heads, mlp_ratio, qkv_bias, p, attn_p)
        self.cross_atten_block3 = cross_atten(self.patch_size, embed_dim, self.num_patches, depth_self,
                                              depth_cross, num_heads, mlp_ratio, qkv_bias, p, attn_p)

    def forward(self, x1, x2, shift_flag=False):

        x1_1, x2_1, paddings1 = self.self_atten_block1(x1, x2)
        out1, x1_self1, x2_self1, x1_cross1, x2_cross1 = self.cross_atten_block1(x1_1, x2_1, paddings1)

        x1_2, x2_2, paddings2 = self.self_atten_block2(x1_cross1, x2_cross1)
        out2, x1_self2, x2_self2, x1_cross2, x2_cross2 = self.cross_atten_block2(x1_2, x2_2, paddings2)
        # out = out1 + out2
        x1_3, x2_3, paddings3 = self.self_atten_block3(x1_cross2, x2_cross2)
        out3, x1_self3, x2_self3, x1_cross3, x2_cross3 = self.cross_atten_block3(x1_3, x2_3, paddings3)
        #
        out = out1 + out2 + out3
        x_cross1, x_cross2 = x1_cross3, x2_cross3

        return out, x1_1, x2_1, x_cross1, x_cross2

