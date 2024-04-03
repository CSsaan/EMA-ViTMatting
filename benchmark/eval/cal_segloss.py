import torch

class IoULoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        pred = pred > 0.5
        target = target > 0.5
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
        pred = pred > 0.5
        target = target > 0.5
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice_coeff = (2. * intersection + self.eps) / (union + self.eps)
        loss = 1 - dice_coeff
        return loss
    
if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    predict = torch.randn(1, 1, 320, 320)
    alpha = predict
    # alpha = torch.randn(1, 1, 320, 320)

    # predict = torch.zeros(1, 1, 320, 320)
    # alpha = torch.ones(1, 1, 320, 320)
    
    # 实例化 DiceLoss、IoULoss 类
    Dice = DiceLoss().to(predict.device)
    IoU = IoULoss().to(predict.device)

    # 计算损失大小
    loss = Dice(predict, alpha)
    print("Dice: {:.6f}".format(loss.item()))

    loss = IoU(predict, alpha)
    print("IoU: {:.6f}".format(loss.item()))