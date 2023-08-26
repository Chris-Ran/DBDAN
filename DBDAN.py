import torch
import torch.nn as nn
import torch.nn.functional as F
from hrnet import get_hrnetv2_w32


def ds_conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def patch_split(input, patch_size):
    """
    input: (B, C, H, W)
    output: (B*num_h*num_w, C, patch_h, patch_w)
    """
    B, C, H, W = input.size()
    patch_h, patch_w = patch_size
    num_h, num_w = H // patch_h, W // patch_w
    out = input.view(B, C, num_h, patch_h, num_w, patch_w)
    out = out.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, patch_h, patch_w)
    return out


def patch_recover(input, img_size):
    """
    input: (B*num_h*num_w, C, patch_h, patch_w)
    output: (B, C, H, W)
    """
    N, C, patch_h, patch_w = input.size()
    H, W = img_size
    num_h, num_w = H // patch_h, W // patch_w
    B = N // (num_h * num_w)
    out = input.view(B, num_h, num_w, C, patch_h, patch_w)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return out


class CorrelationAttention_local(nn.Module):
    def __init__(self, feat_size, match_size):
        super(CorrelationAttention_local, self).__init__()
        self.feat_size = feat_size
        self.match_size = match_size
        self.patch_corr_proj = nn.Linear(self.feat_size, self.match_size)
        self.feat_norm1 = nn.LayerNorm(self.match_size)
        self.feat_norm2 = nn.LayerNorm(self.feat_size)
        self.out_patch_corr_proj = nn.Linear(self.feat_size, self.feat_size)

    def forward(self, patch_corr_map):
        patch_corr_map_proj = self.patch_corr_proj(patch_corr_map)
        patch_corr_map_norm = self.feat_norm1(patch_corr_map_proj)

        q = k = patch_corr_map_norm
        v = self.feat_norm2(patch_corr_map)

        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)

        patch_corr_map = attn @ v
        patch_corr_map = patch_corr_map + self.out_patch_corr_proj(patch_corr_map)
        return patch_corr_map


class LocalDynamicAttention(nn.Module):
    def __init__(self, dim, feat_size=1024):
        super(LocalDynamicAttention, self).__init__()
        self.dim = dim
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.corr_attn = CorrelationAttention_local(
            dropout=0.0, feat_size=feat_size, match_size=64
        )

    def forward(self, _x):
        x = patch_split(_x, (32, 32))
        b, c, h, w = x.shape
        residual = x
        q = self.q_proj(x).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        k = self.k_proj(x).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        v = self.v_proj(x).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()

        q = q * (c**-0.5)
        attn = q @ k.transpose(-2, -1)  # (b,n,c) * (b,c,n) -> (b,n,n)

        corr_map = attn
        corr_map = self.corr_attn(corr_map)
        attn = attn + corr_map

        attn = F.softmax(attn, dim=-1)

        x = attn @ v  # (b,n,n) * (b,n,c) -> (b,n,c)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x = self.out_proj(x)

        out = patch_recover(x + residual, (x.size(2), x.size(3)))

        return out


def AvgPooling(input, out_size):
    pool = nn.AdaptiveAvgPool2d(output_size=out_size)
    out = pool(input)
    return out


class CorrelationAttention_global(nn.Module):
    def __init__(self, feat_size, match_size):
        super(CorrelationAttention_global, self).__init__()
        self.feat_size = feat_size
        self.match_size = match_size

    def forward(self, patch_corr_map):
        q = k = v = patch_corr_map

        attn = q @ k.transpose(-2, -1)  # (b,n,s) * (b,s,n) -> (b,n,n)
        attn = F.softmax(attn, dim=-1)

        patch_corr_map = attn @ v  # (b,n,n) * (b,n,s) -> (b,n,s)
        return patch_corr_map


class GlobalDynamicAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalDynamicAttention, self).__init__()
        self.dim = dim
        self.q_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.corr_attn = CorrelationAttention_global(dropout=0.0, feat_size=1024, match_size=64)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        q = patch_split(x, (32, 32))
        q = self.q_proj(q).reshape(-1, c // 2, 1024).permute(0, 2, 1).contiguous()

        k = self.k_proj(x)
        k = AvgPooling(k, (h // 32, w // 32)).reshape(b, c // 2, -1)  # (b,c,num_h*num_w)
        _, _, num = k.shape
        v = self.v_proj(x)
        v = (
            AvgPooling(v, (h // 32, w // 32))
            .reshape(b, c // 2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )  # (b,num_h*num_w,c)

        q = q * (c**-0.5)
        k = k.repeat(num, 1, 1)
        attn = q @ k  # (b,n,c) * (b,c,s) -> (b,n,s)

        corr_map = attn
        corr_map = self.corr_attn(corr_map)
        attn = attn + corr_map

        attn = F.softmax(attn, dim=-1)

        v = v.repeat(num, 1, 1)
        out = attn @ v
        out = out.permute(0, 2, 1).contiguous().reshape(-1, c // 2, 32, 32)
        out = patch_recover(out, (x.size(2), x.size(3)))
        out = self.out_proj(out)

        return out + residual


class AB(nn.Module):
    def __init__(self, in_channel, reduction=16, K=2, t=0.5):
        super(AB, self).__init__()
        self.t = t
        self.K = K

        self.conv_last = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, self.K, bias=False),
        )

        # local_dynamic_attention branch
        self.local_dynamic_attention = LocalDynamicAttention(in_channel)

        # non-attention branch
        self.global_dynamic_attention = GlobalDynamicAttention(dim=in_channel)

    def forward(self, x):
        residual = x
        a, b, _, _ = x.shape

        # Dynamic Weighting Allocation Mechanism
        y = self.avg_pool(x).view(a, b)
        y = self.ADM(y)
        ax = F.softmax(y / self.t, dim=1)

        # Local Dynamic Attention Branch & Global Dynamic Attention Branch
        local_dynamic_attention = self.local_dynamic_attention(x)
        global_dynamic_attention = self.global_dynamic_attention(x)

        x = local_dynamic_attention * ax[:, 0].view(a, 1, 1, 1) + global_dynamic_attention * ax[
            :, 1
        ].view(a, 1, 1, 1)
        x = self.lrelu(x)

        out = self.conv_last(x)
        out += residual

        return out


class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=6, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False
        )
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view(
            (N, H, W * self.scale, int(C / (self.scale)))
        )

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale)))
        )

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)

        return x


class DBDAN(nn.Module):
    def __init__(self, num_classes):
        super(DBDAN, self).__init__()
        self.num_classes = num_classes
        self.backbone = get_hrnetv2_w32()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.AB = AB(in_channel=512)
        self.attn_conv = ds_conv(
            in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True
        )

        self.cat_conv = ds_conv(512 + 480, 512, 3, padding=1, bias=True)

        self.dsn = nn.Sequential(
            nn.Conv2d(480, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.05),
        )
        self.upsample = DUpsampling(inplanes=512, scale=4, num_class=num_classes)

    def forward(self, x_):
        feat = self.backbone(x_)[-1]
        aux = feat

        x = self.bottleneck(feat)
        x_attn = self.attn_conv(self.AB(x))
        x = x + x_attn

        x = self.cat_conv(torch.cat((x, feat), dim=1))
        x = self.upsample(x)

        if self.training:
            aux = self.upsample(self.dsn(aux))
            return x, aux
        else:
            return x


if __name__ == "__main__":
    model = DBDAN(num_classes=6)
    x = torch.randn(4, 3, 512, 512)
    outs = model(x)
    for out in outs:
        print(out.shape)
