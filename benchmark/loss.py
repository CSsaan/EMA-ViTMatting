import torch
import torch.nn.functional as F
import kornia
from .eval import LapLoss, IoULoss, DiceLoss


class MattingLoss(torch.nn.Module):
    def __init__(self):
        super(MattingLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Lap = LapLoss(max_levels=2).to(device)
        self.IoU = IoULoss().to(device)
        self.Dice = DiceLoss().to(device)
        # 创建一个3x3的Laplacian算子
        self.laplacian_kernel = kornia.filters.get_laplacian_kernel2d(kernel_size=(3,3))

    def mse_loss(self, predict, alpha):
        weighted = torch.ones(alpha.shape).to(predict.device)
        alpha_f = alpha.to(predict.device)
        diff = predict - alpha_f
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())
        return alpha_loss
    
    def composition_loss(self, predict, alpha, img):
        fg = torch.cat((alpha, alpha, alpha), 1) * img
        fg_pre = torch.cat((predict, predict, predict), 1) * img
        eps = 1e-6
        L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()
        return L_composition

    def forward(self, predict, alpha, img):
        # _mse = self.mse_loss(predict, alpha)
        
        Lap_loss = self.Lap(predict, alpha)
        iou_loss = self.IoU(predict, alpha)
        dice_loss = self.Dice(predict, alpha)
        l1 = F.l1_loss(predict, alpha)
        l1_sobel = F.l1_loss(kornia.filters.laplacian(predict, 3), kornia.filters.laplacian(alpha, 3))
        mse = F.mse_loss(predict, alpha)
        # L_composition
        com_loss = self.composition_loss(predict, alpha, img)
        loss_all = (l1 + mse) + 2.0*l1_sobel + (0.5*Lap_loss + iou_loss + dice_loss) + com_loss

        loss_dict = {
            'l1_loss': l1.item(),
            'mse_loss': mse.item(),
            'l1_sobel_loss': 2.0 * l1_sobel.item(),
            'laplacian_loss': 0.5 * Lap_loss.item(),
            'iou_loss': iou_loss.item(),
            'dice_loss': dice_loss.item(),
            'com_loss': com_loss.item(),
            'loss_all': loss_all
        }
        return loss_dict

if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    img = torch.randn(1, 3, 320, 320)

    predict = torch.randn(1, 1, 320, 320)
    alpha = predict
    # alpha = torch.randn(1, 1, 320, 320)

    # predict = torch.zeros(1, 1, 320, 320)
    # alpha = torch.ones(1, 1, 320, 320)
    
    # 实例化 MattingLoss 类
    matting_loss = MattingLoss().to(predict.device)

    # 计算损失大小
    loss = matting_loss(predict, alpha, img)
    print("{:.6f}".format(loss.item()))