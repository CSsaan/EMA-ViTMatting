import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# netron模型可视化
import netron
import torch.onnx
import onnx
from onnxsim import simplify

class VGG(nn.Module):
    def __init__(self, in_channels):
        super(VGG, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=1), # (1, 224, 224) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), # (64, 224, 224) -> (64, 224, 224)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (64, 224, 224) -> (64, 112, 112)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), # (64, 112, 112) -> (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), # (128, 112, 112) -> (128, 112, 112)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (128, 112, 112) -> (128, 56, 56)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), # (128, 56, 56) -> (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), # (256, 56, 56) -> (256, 56, 56)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (256, 56, 56) -> (256, 28, 28)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), # (256, 56, 56) -> (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), # (512, 28, 28) -> (512, 28, 28)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (512, 28, 28) -> (512, 14, 14)
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), # (512, 14, 14) -> (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), # (512, 14, 14) -> (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # (512, 14, 14) -> (512, 7, 7)
        )
        self.b6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 256), # (512, 7, 7) -> 25088
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

# ---------------------------------------------------------------------------------------
def summary_model(model, input_x):
    print(summary(model, input_x))

def ONNX_model(model, input_x):
    #导出模型
    save_origion = False
    if(save_origion):
        modelData = "./onnx/VGG_origion.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
        torch.onnx.export(model, input_x, modelData, opset_version = 11)  # 将 pytorch 模型以 onnx 格式导出并保存
        netron.start(modelData)  # 输出网络结构
    else:
        # 去除identy层（简化模型）
        modelData = "./onnx/VGG_origion.onnx"
        SimpledData = "./onnx/VGG_simplified.onnx"
        torch.onnx.export(model, input_x, modelData, opset_version = 11)
        onnx_model = onnx.load(modelData)
        simplified_model, check = simplify(onnx_model)
        onnx.save_model(simplified_model, SimpledData)
        netron.start(SimpledData)  # 输出网络结构

# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(in_channels = 1).to(device)

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
