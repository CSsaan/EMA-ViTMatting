# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from typing import Callable, Optional

from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



if __name__ == "__main__":
    # netron模型可视化
    import netron
    import torch.onnx
    from torch.autograd import Variable
    from torchsummary import summary
    
    # 打印模型网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mlp(in_features=768, hidden_features=768*2).to(device)
    print(summary(model, (196, 768))) # B,N,D: -1,196,768

    # 可视化模型网络结构
    input_x = torch.randn(8, 196, 768)  # 随机生成一个输入
    # modelData = "./demo.pth"  # 定义模型数据保存的路径
    modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
    torch.onnx.export(model, input_x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    # netron.start(modelData)  # 输出网络结构