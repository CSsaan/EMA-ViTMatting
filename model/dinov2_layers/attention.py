# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    """
    qkv Attention: (B,N,D) -> (B,N,D)
    (   
        B: Batch Size;
        N: Num Patches;
        D: Embed Dim.
    )
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 扩充出3个维度（用来后续转为q\k\v）
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # qkv():     [ B, N, C ]                                 -->  [ B, N, 3*C ]
        # reshape(): [ B, N, 3*C ]                               -->  [ B, N, 3, num_heads, embed_dim_per_head ]
        # permute(): [ B, N, 3, num_heads, embed_dim_per_head ]  -->  [ 3, B, num_heads, N, embed_dim_per_head ]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # transpose(): [ B, num_heads, N, embed_dim_per_head ]  -->  [B, num_heads, embed_dim_per_head, N]
        # @: multiply -> [B, num_heads, N, N]
        q, k, v = qkv[0]*self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



if __name__ == "__main__":
    # netron模型可视化
    import netron
    import torch.onnx
    from torch.autograd import Variable
    from torchsummary import summary
    
    # 打印模型网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemEffAttention(dim=768, num_heads=8).to(device)
    print(summary(model, (196, 768))) # B,N,D: 8, 196, 768

    # 可视化模型网络结构
    input_x = torch.randn(8, 196, 768)  # 随机生成一个输入
    # modelData = "./demo.pth"  # 定义模型数据保存的路径
    modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
    torch.onnx.export(model, input_x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构