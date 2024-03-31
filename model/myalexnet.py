import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# netron模型可视化
import netron
import torch.onnx
import onnx
from onnxsim import simplify

class AlexNet(torch.nn.Module):
    def __init__(self, in_channels):
        super(AlexNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, padding=0, stride=4), # (1, 227, 227) -> (96, 55, 55)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2), # (96, 55, 55) -> (96, 27, 27)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=3, padding=1, stride=1), # (96, 27, 27) -> (256, 27, 27)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2), # (256, 27, 27) -> (256, 13, 13)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1), # (256, 13, 13) -> (384, 13, 13)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1), # (384, 13, 13) -> (384, 13, 13)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1), # (384, 13, 13) -> (256, 13, 13)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2), # (256, 13, 13) -> (256, 6, 6)
        )
        self.liner = nn.Sequential(
            nn.Linear(9216, 4096), # (1, 9216) -> 4096
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), # 4096 -> 4096
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10), # 4096 -> 10
        )
        self.seq = nn.Sequential(
            self.down1,   # (1, 227, 227) -> (96, 27, 27)
            self.down2,   # (96, 27, 27) -> (256, 13, 13)
            self.conv1,    # (256, 13, 13) -> (384, 13, 13)
            self.conv2,    # (384, 13, 13) -> (384, 13, 13)
            self.down3,   # (384, 13, 13) -> (256, 6, 6)
            nn.Flatten(), # (256, 6, 6) -> 9216
            self.liner,   # 9216 -> 10
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.seq(x)


# ---------------------------------------------------------------------------------------
def summary_model(model, input_x):
    print(summary(model, input_x))

def ONNX_model(model, input_x):
    #导出模型
    save_origion = False
    if(save_origion):
        modelData = "./onnx/AlexNet_origion.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
        torch.onnx.export(model, input_x, modelData, opset_version = 11)  # 将 pytorch 模型以 onnx 格式导出并保存
        netron.start(modelData)  # 输出网络结构
    else:
        # 去除identy层（简化模型）
        modelData = "./onnx/AlexNet_origion.onnx"
        SimpledData = "./onnx/AlexNet_simplified.onnx"
        torch.onnx.export(model, input_x, modelData, opset_version = 11)
        onnx_model = onnx.load(modelData)
        simplified_model, check = simplify(onnx_model)
        onnx.save_model(simplified_model, SimpledData)
        netron.start(SimpledData)  # 输出网络结构

# ---------------------------------------------------------------------------------------
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(in_channels = 1).to(device)

    # (1). summary打印模型网络结构
    input_x = (1, 227, 227)
    summary_model(model, input_x)

    # (2). ONNX可视化模型网络结构
    input_x = torch.randn(8, 1, 227, 227)
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