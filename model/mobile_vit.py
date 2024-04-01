import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce

# 模型可视化
from torchsummary import summary
import torch.nn.functional as F

# helpers

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        # print(f"out.size:{out.size()}")
        return out

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.1):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape # [1, 96, 32, 32]
        # print(f"0:{x.size()}")
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print(f"1:{x.size()}")
        x = self.transformer(x)
        # print(f"2:{x.size()}")
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw) # [1, 96, 32, 32]
        # print(f"3:{x.size()}")
        # Fusion
        x = self.conv3(x) # [1, 64, 32, 32]
        # print(f"4:{x.size()}")
        x = torch.cat((x, y), 1) # [1, 128, 32, 32]
        # print(f"5:{x.size()}")
        x = self.conv4(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.silu(x)
        return x

class UpsampleBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(UpsampleBlock2, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        self.re0 = None
        self.re1 = None
        self.re2 = None

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        # ---------------- upscale -------------------
        self.trunk_up = nn.ModuleList([])
        self.trunk_up.append(
            nn.ConvTranspose2d(channels[9], channels[7], kernel_size=2, stride=2, padding=0),
        )
        self.trunk_up.append(
            nn.ConvTranspose2d(channels[7], channels[5], kernel_size=2, stride=2, padding=0),
        )
        self.trunk_up.append(
            nn.ConvTranspose2d(channels[5], channels[3], kernel_size=2, stride=2, padding=0),
        )
        self.trunk_up.append(
            nn.ConvTranspose2d(channels[3], channels[1], kernel_size=2, stride=2, padding=0),
        )
        self.trunk_up.append(
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, padding=0),
        )

        # self.up1 = UpsampleBlock(channels[9], channels[7])
        # self.up2 = UpsampleBlock(channels[7], channels[5])
        # self.up3 = UpsampleBlock(channels[5], channels[3])
        # self.up4 = UpsampleBlock(channels[3], channels[1])
        # self.up5 = UpsampleBlock(channels[1], channels[0])

        self.up1 = UpsampleBlock2(channels[9], channels[7], bias=False)
        self.up2 = UpsampleBlock2(channels[7], channels[5], bias=False)
        self.up3 = UpsampleBlock2(channels[5], channels[3], bias=False)
        self.up4 = UpsampleBlock2(channels[3], channels[1], bias=False)
        self.up5 = UpsampleBlock2(channels[1], channels[0])

        self.to_logits = nn.Sequential(
            Reduce('b c h w -> b h w', 'mean'),
        )

    def forward(self, x):
        x = self.conv1(x) # [1, 3, 256, 256] -> [1, 16, 128, 128]
        for idx, conv in enumerate(self.stem):
            x = conv(x) # [1, 32, 128, 128] [1, 48, 64, 64] [1, 48, 64, 64] [1, 48, 64, 64]
            if idx == 0:
                self.re128 = x
        self.re64 = x
        for idx, (conv, attn) in enumerate(self.trunk):
            x = conv(x) # [1, 64, 32, 32] [1, 80, 16, 16] [1, 96, 8, 8]
            x = attn(x) # (与上面同尺寸)
            # print(idx, " : ",x.size())
            if idx == 0:
                self.re32 = x
            elif idx == 1:
                self.re16 = x


        # 反向编码上采样
        x = self.up1(x) + self.re16 # [8, 96, 8, 8]  ->  [1, 80, 16, 16] [1, 64, 32, 32] [1, 48, 64, 64] [1, 32, 128, 128] [1, 256, 256]
        x = self.up2(x) + self.re32
        x = self.up3(x) + self.re64
        x = self.up4(x) + self.re128
        x = self.up5(x)

        return self.to_logits(x)
        return x


def summary_model(model, input_x):
    print(summary(model, input_x))

if __name__ == '__main__':
    mbvit_xs = MobileViT(
        image_size = (256, 256),
        dims = [144, 180, 216],
        channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96],
        depths = (8, 16, 12)
    )
    img = torch.randn(8, 3, 256, 256)
    pred = mbvit_xs(img) # (1, 1000)
    print(f"pred:{pred.size()}")


    # # (1). summary打印模型网络结构
    # input_x = (3, 256, 256)
    # summary_model(mbvit_xs, input_x)