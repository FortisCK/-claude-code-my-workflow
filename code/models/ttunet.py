"""ttunet.py — faithful port of TT U-Net (Deng et al., IEEE TMI 2023).

Temporal-Transformer U-Net for cardiac CT motion-artifact reduction. Ported from the
authors' reference notebook (external/TT-U-Net/code/TTUNet/TTUNet_demo.ipynb) for use as
a published BASELINE on our ImageCAS data.

Faithful-adaptation notes (documented in
quality_reports/plans/2026-06-21_ttunet-baseline-reproduction.md):
  - Input is (B, 1, S, H, W); in the original S = cardiac-phase frames (hardcoded 48).
    Our data are single 3D volumes -> we treat the z-axis (depth) as the temporal
    sequence S. The temporal transformer then attends across axial slices.
  - The hardcoded S=48 and per-level spatial grid [64,64]/[32,32]/[16,16] (for 256^2 input)
    are made DYNAMIC so the net runs on arbitrary patches (spatial divisible by 16).
  - Architecture, channel widths (cnum=24), and residual output are unchanged.

timm helpers (DropPath / trunc_normal_ / to_2tuple) are inlined to avoid a dependency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_


def _to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep) * mask.floor_()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


# ---- conv blocks (3D, leaky-relu) ----
class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class UpConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.lrelu(self.conv(x))


class DeConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class ConvAct2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.conv(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    """Multi-head self-attention over the temporal tokens (window_size=1 spatial)."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        ).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Temporal self-attention block; attends along S (passed at runtime)."""

    def __init__(self, dim, num_heads, window_size=1, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, _to_2tuple(window_size), num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, s_len: int):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x.view(-1, s_len, C)).view(B, L, C)  # attention over the S (depth) tokens
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Spatial /2 conv then flatten spatial+temporal to tokens (S kept as innermost)."""

    def __init__(self, in_chans):
        super().__init__()
        self.conv = nn.Conv3d(in_chans, 4 * in_chans, kernel_size=[1, 3, 3], padding=[0, 1, 1], stride=[1, 2, 2])

    def forward(self, x):
        x = self.conv(x)
        B, C, S, H, W = x.shape
        return x.permute(0, 3, 4, 2, 1).reshape(B, -1, C)  # (B, H*W*S, C), S innermost


class PatchUnEmbed(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_chans, int(in_chans / 4), [1, 3, 3], padding=[0, 1, 1], stride=[1, 1, 1])

    def forward(self, x, x_size):
        B, _, C = x.shape
        x = x.view(B, x_size[0], x_size[1], -1, C).permute(0, 4, 3, 1, 2)  # B,C,S,H,W
        x = F.interpolate(x, scale_factor=[1, 2, 2], mode="nearest")
        return self.conv(x)


class TTUNet(nn.Module):
    """TT U-Net generator (Deng et al. 2023). Residual output y = f(x) + x."""

    def __init__(self, cnum: int = 24):
        super().__init__()
        c = cnum
        self.conv1 = ConvAct(1, c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.conv1_2 = ConvAct(c, c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.convt1_1 = ConvAct(c, c, [5, 1, 1], [2, 0, 0], [1, 1, 1])
        self.convt1_2 = ConvAct(c, c, [5, 1, 1], [2, 0, 0], [1, 1, 1])
        self.convt1_3 = ConvAct(c, c, [5, 1, 1], [2, 0, 0], [1, 1, 1])
        self.conv2 = ConvAct(c, 2 * c, [1, 3, 3], [0, 1, 1], [1, 2, 2])
        self.conv3 = ConvAct(2 * c, 2 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.conv3_2 = ConvAct(2 * c, 2 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.embed1 = PatchEmbed(2 * c)
        self.trans1_1 = TransformerBlock(8 * c, num_heads=c * 32 // 16, drop_path=0.05)
        self.trans1_2 = TransformerBlock(8 * c, num_heads=c * 32 // 16, drop_path=0.05)
        self.unembed1 = PatchUnEmbed(8 * c)
        self.convt1 = ConvAct(2 * c, 2 * c, [3, 1, 1], [1, 0, 0], [1, 1, 1])
        self.conv4 = ConvAct(2 * c, 4 * c, [1, 3, 3], [0, 1, 1], [1, 2, 2])
        self.conv5 = ConvAct(4 * c, 4 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.conv5_2 = ConvAct(4 * c, 4 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.embed2 = PatchEmbed(4 * c)
        self.trans2_1 = TransformerBlock(16 * c, num_heads=c * 16 // 16, drop_path=0.1)
        self.trans2_2 = TransformerBlock(16 * c, num_heads=c * 16 // 16, drop_path=0.1)
        self.unembed2 = PatchUnEmbed(16 * c)
        self.convt2 = ConvAct(4 * c, 4 * c, [3, 1, 1], [1, 0, 0], [1, 1, 1])
        self.conv6 = ConvAct(4 * c, 8 * c, [1, 3, 3], [0, 1, 1], [1, 2, 2])
        self.conv7 = ConvAct(8 * c, 8 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.conv7_2 = ConvAct(8 * c, 8 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.embed3 = PatchEmbed(8 * c)
        self.trans3_1 = TransformerBlock(32 * c, num_heads=c * 32 // 16, drop_path=0.1)
        self.trans3_2 = TransformerBlock(32 * c, num_heads=c * 32 // 16, drop_path=0.1)
        self.unembed3 = PatchUnEmbed(32 * c)
        self.convt3 = ConvAct(8 * c, 8 * c, [3, 1, 1], [1, 0, 0], [1, 1, 1])
        self.conv8 = DeConvAct(8 * c, 8 * c, [1, 3, 3], [0, 1, 1], [1, 1, 1])
        self.conv9 = UpConvAct(8 * c, 4 * c, [1, 3, 3], 1, [0, 1, 1], scale_factor=[1, 2, 2])
        self.conv10 = DeConvAct(8 * c, 4 * c, 3, 1, 1)
        self.conv10_2 = DeConvAct(4 * c, 4 * c, 3, 1, 1)
        self.conv11 = UpConvAct(4 * c, 2 * c, [1, 3, 3], 1, [0, 1, 1], scale_factor=[1, 2, 2])
        self.conv12 = DeConvAct(4 * c, 2 * c, 3, 1, 1)
        self.conv12_2 = DeConvAct(2 * c, 2 * c, 3, 1, 1)
        self.conv13 = UpConvAct(2 * c, 1 * c, [1, 3, 3], 1, [0, 1, 1], scale_factor=[1, 2, 2])
        self.conv14 = DeConvAct(2 * c, 1 * c, 3, 1, 1)
        self.conv15 = DeConvAct(1 * c, 1 * c, 3, 1, 1)
        self.conv16 = nn.Conv3d(c, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[2]  # temporal length (= z-depth in our adaptation)
        x0 = self.conv1_2(self.conv1(x))
        x0_t = self.convt1_3(self.convt1_2(self.convt1_1(x0)))
        x1 = self.conv3_2(self.conv3(self.conv2(x0)))
        s1 = (x1.shape[-2] // 2, x1.shape[-1] // 2)
        x1_t = self.convt1(self.unembed1(self.trans1_2(self.trans1_1(self.embed1(x1), s), s), s1)) + x1
        x2 = self.conv5_2(self.conv5(self.conv4(x1)))
        s2 = (x2.shape[-2] // 2, x2.shape[-1] // 2)
        x2_t = self.convt2(self.unembed2(self.trans2_2(self.trans2_1(self.embed2(x2), s), s), s2)) + x2
        x3 = self.conv7_2(self.conv7(self.conv6(x2)))
        s3 = (x3.shape[-2] // 2, x3.shape[-1] // 2)
        x3_t = self.convt3(self.unembed3(self.trans3_2(self.trans3_1(self.embed3(x3), s), s), s3)) + x3
        x4 = self.conv9(self.conv8(x3_t))
        x5 = self.conv11(self.conv10_2(self.conv10(torch.cat([x4, x2_t], dim=1))))
        x6 = self.conv13(self.conv12_2(self.conv12(torch.cat([x5, x1_t], dim=1))))
        y = self.conv16(self.conv15(self.conv14(torch.cat([x6, x0_t], dim=1)))) + x
        return y


class Dis(nn.Module):
    """2D PatchGAN discriminator (operates on axial slices). Optional adversarial term."""

    def __init__(self, cnum: int = 24):
        super().__init__()
        self.conv1 = ConvAct2D(1, cnum, 5, 2, 2)
        self.conv2 = ConvAct2D(cnum, cnum * 2, 5, 2, 2)
        self.conv3 = ConvAct2D(cnum * 2, cnum * 4, 5, 2, 2)
        self.conv4 = ConvAct2D(cnum * 4, cnum * 8, 4, 1, 1)
        self.conv5 = nn.Conv2d(cnum * 8, 1, 4, 1, 1)

    def forward(self, x):
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
