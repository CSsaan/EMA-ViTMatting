import cv2
import torch

class IoULoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred > 0.
        target = target > 0.
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = (intersection + self.eps) / (union + self.eps)
        loss = 1 - iou
        return loss

class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred > 0.
        target = target > 0.
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice_coeff
        return loss
    
if __name__ == '__main__':
    # opencv加载本地灰度图像,并转为为tensor格式的(1,1,w,h)形状
    alpha = cv2.imread('data/AIM500/test/mask/o_dc288b1a.png', cv2.IMREAD_GRAYSCALE)
    # 调整形状为(1, 1, H, W)
    alpha = torch.from_numpy(alpha).unsqueeze(0).unsqueeze(0).float()
    predict = alpha.clone()
    alpha[:, :, :predict.size()[3]//4, :] = 0
    
    # 实例化 DiceLoss、IoULoss 类
    Dice = DiceLoss().to(predict.device)
    IoU = IoULoss().to(predict.device)

    # 计算损失大小
    loss = Dice(predict, alpha)
    print("Dice: {:.6f}".format(loss.item()))

    loss = IoU(predict, alpha)
    print("IoU: {:.6f}".format(loss.item()))