import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# netron模型可视化
import netron
import torch.onnx
import onnx
from onnxsim import simplify


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, UseRes=False):
        super(Residual, self).__init__()
        self.UseRes = UseRes
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        out = self.b1(x)
        if(self.UseRes):
            out = out + self.b2(x)
        return out

class ResNet(nn.Module):
    def __init__(self, Residual, in_channels):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2), # (1, 224, 224) -> (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2), # (64, 112, 112) -> (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, 1, False), # (64, 56, 56) -> (64, 56, 56)
            Residual(64, 64, 1, False), # (64, 56, 56) -> (64, 56, 56)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, 2, True), # (64, 56, 56) -> (128, 28, 28)
            Residual(128, 128, 1, False), # (128, 28, 28) -> (128, 28, 28)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, 2, True), # (128, 28, 28) -> (256, 14, 14)
            Residual(256, 256, 1, False), # (256, 14, 14) -> (256, 14, 14)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, 2, True), # (256, 14, 14) -> (512, 7, 7)
            Residual(512, 512, 1, False), # (512, 7, 7) -> (512, 7, 7)
        )
        self.b6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 10)
        )

    def forward(self, x):
        x = self.b1(x) # (1, 224, 224) -> (64, 56, 56)
        x = self.b2(x) # (1, 224, 224) -> (64, 56, 56)
        x = self.b3(x) # (64, 56, 56) -> (128, 28, 28)
        x = self.b4(x) # (128, 28, 28) -> (256, 14, 14)
        x = self.b5(x) # (256, 14, 14) -> (512, 7, 7)
        x = self.b6(x) # (512, 7, 7) -> 25088 -> 10
        return x


# ---------------------------------------------------------------------------------------
def summary_model(model, input_x):
    print(summary(model, input_x))

def ONNX_model(model, input_x):
    #导出模型
    save_origion = False
    if(save_origion):
        modelData = "./onnx/ResNet_origion.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
        torch.onnx.export(model, input_x, modelData, opset_version = 11)  # 将 pytorch 模型以 onnx 格式导出并保存
        netron.start(modelData)  # 输出网络结构
    else:
        # 去除identy层（简化模型）
        modelData = "./onnx/ResNet_origion.onnx"
        SimpledData = "./onnx/ResNet_simplified.onnx"
        torch.onnx.export(model, input_x, modelData, opset_version = 11)
        onnx_model = onnx.load(modelData)
        simplified_model, check = simplify(onnx_model)
        onnx.save_model(simplified_model, SimpledData)
        netron.start(SimpledData)  # 输出网络结构

# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(Residual, in_channels = 1).to(device)

    # (1). summary打印模型网络结构
    input_x = (1, 224, 224)
    summary_model(model, input_x)

    # (2). ONNX可视化模型网络结构
    input_x = torch.randn(8, 1, 224, 224)
    ONNX_model(model, input_x)


"""
(W-k+2p)/s + 1

keepsize:( 理解为括号内为(W-1),s=1 )
    k  p    s
    1  0 \      :(32-1+2*0)/1+1 = 32 (一般用来增减通道数)
    3  1 — [s1] :(32-3+2*1)/1+1 = 32
    5  2  /     :(32-5+2*2)/1+1 = 32
    7  3 /      :(32-7+3*2)/1+1 = 32
    9  4/       :(32-9+4*2)/1+1 = 32


dowmscale:( 理解为括号内为(W-1),s=2 )
 原大小为偶数:
    k  p    s
    3  1 \      :(32-1+2*0)/2+1 = 16
    5  2 — [s2] :(32-5+2*2)/2+1 = 16
    7  3 /      :(32-7+2*3)/2+1 = 16
    9  4/       :(32-9+4*2)/2+1 = 16
    MaxPoolxd AvgPoolxd
    AdaptiveAvgPoolxd
 原大小为基数:
    满足p = 0


upscale:
    nn.UpsamplingBilinear2d(scale_factor = 2)
    nn.PixelShuffle(2)

"""
