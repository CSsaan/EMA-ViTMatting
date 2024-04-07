import torch
import torch.nn.functional as F
import kornia
import cv2

import numpy as np
import os
import sys
# 获取当前文件所在目录的上2级目录
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(config_path)
from eval import LapLoss, IoULoss, DiceLoss


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
    
    def laplacian_edge_detection(self, img_tensor, kernel_size=3):
        laplacian = kornia.filters.laplacian(img_tensor, kernel_size=kernel_size)/255.0*2.0
        # edge = laplacian.squeeze().numpy()
        # cv2.imshow('edge', edge)
        # cv2.waitKey(0)
        return laplacian
    
    def mask_loss(self, predict, alpha):
        non_zero_indices = alpha.nonzero()
        non_zero_predict = predict[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]
        non_zero_alpha = alpha[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2], non_zero_indices[:, 3]]
        l1_loss = torch.mean(torch.abs(non_zero_predict - non_zero_alpha))
        return l1_loss

    def forward(self, predict, alpha, img):
        # _mse = self.mse_loss(predict, alpha)
        
        Lap_loss = self.Lap(predict, alpha)
        iou_loss = self.IoU(predict, alpha)
        dice_loss = self.Dice(predict, alpha)
        l1 = F.l1_loss(predict, alpha)
        l1_sobel = F.l1_loss(self.laplacian_edge_detection(predict, 3), self.laplacian_edge_detection(alpha, 3))
        mse = F.mse_loss(predict, alpha)
        # L_composition
        com_loss = self.composition_loss(predict, alpha, img)
        mask_L1_loss = self.mask_loss(predict, alpha)
        loss_all = (l1 + mse)*0.5 + (iou_loss + dice_loss) + Lap_loss + mask_L1_loss # (l1 + mse)*10.0 + (iou_loss + dice_loss)*100.0 + Lap_loss + mask_L1_loss # (l1 + mse)*10.0 + 200.0*l1_sobel + (Lap_loss + iou_loss + dice_loss) + com_loss

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
    img = torch.randn(1, 3, 2048, 1365)
    img = F.interpolate(img, size=(64, 64), mode='bilinear', align_corners=False)

    # predict = torch.zeros(1, 1, 2048, 1365)
    predict = cv2.imread('/workspaces/EMA-ViTMatting/data/AIM500/train/mask/o_c2cf00c3.png', cv2.IMREAD_GRAYSCALE)
    predict = kornia.image_to_tensor(predict).float().unsqueeze(0)
    predict = F.interpolate(predict, size=(64, 64), mode='bilinear', align_corners=False)
    predict = torch.clamp(predict, min=0.0, max=1.0)
    # print(predict.size())
    alpha = predict

    # alpha = torch.ones(1, 1, 2048, 1365)
    alpha = cv2.imread('/workspaces/EMA-ViTMatting/result/o_c2cf00c3.png', cv2.IMREAD_GRAYSCALE)
    alpha = kornia.image_to_tensor(alpha).float().unsqueeze(0)
    alpha = F.interpolate(alpha, size=(64, 64), mode='bilinear', align_corners=False)
    alpha = torch.clamp(alpha, min=0.0, max=1.0)
    # print(alpha.size())

    # 实例化 MattingLoss 类
    matting_loss = MattingLoss().to(predict.device)

    # 计算损失大小
    loss = matting_loss(predict, alpha, img)
    print("{:.6f}".format(loss['loss_all'].item()))
