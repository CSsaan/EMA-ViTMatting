import torch
import torch.nn.functional as F
import kornia
from eval import LapLoss, IoULoss, DiceLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
class MattingLoss(torch.nn.Module):
    def __init__(self):
        super(MattingLoss, self).__init__()

    def mse_loss(self, predict, alpha):
        weighted = torch.ones(alpha.shape).to(predict.device)
        alpha_f = alpha.to(predict.device)
        diff = predict - alpha_f
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())
        return alpha_loss

    def forward(self, predict, alpha):
        # _mse = self.mse_loss(predict, alpha)
        Lap = LapLoss(max_levels=5).to(predict.device)
        IoU = IoULoss().to(predict.device)
        Dice = DiceLoss().to(predict.device)

        Lap_loss = Lap(predict, alpha)
        iou_loss = IoU(predict, alpha)
        dice_loss = Dice(predict, alpha)
        l1 = F.l1_loss(predict, alpha)
        l1_sobel = F.l1_loss(kornia.filters.sobel(predict), kornia.filters.sobel(alpha))
        mse = F.mse_loss(predict, alpha)
        
        return l1 + 2.0*l1_sobel + mse + Lap_loss + iou_loss + dice_loss

if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    predict = torch.randn(1, 1, 320, 320)
    alpha = predict
    # alpha = torch.randn(1, 1, 320, 320)

    # predict = torch.zeros(1, 1, 320, 320)
    # alpha = torch.ones(1, 1, 320, 320)
    
    # 实例化 MattingLoss 类
    matting_loss = MattingLoss().to(predict.device)

    # 计算损失大小
    loss = matting_loss(predict, alpha)
    print("{:.6f}".format(loss.item()))