import torch
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable

# Laplacian pyramid loss

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def pyr_downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def pyr_upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def gauss_kernel5(channels=3, device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]], device=device, dtype=dtype)
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return Variable(kernel, requires_grad=False)


def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    # kernel = gauss_kernel5(n_channels)
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]

def laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    for _level in range(max_levels):
        current = crop_to_even_size(current)
        down = pyr_downsample(current, kernel)
        up = pyr_upsample(down, kernel)

        diff = current - up
        pyr.append(diff)

        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = None

    def forward(self, predict, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != predict.shape[1]:
            self._gauss_kernel = gauss_kernel5(predict.shape[1], device=predict.device, dtype=predict.dtype)

        pyr_predict = laplacian_pyramid_expand(predict, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
        weights = [1, 2, 4, 8, 16]

        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_predict, pyr_target))
        return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_predict, pyr_target))).mean() 
    
if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    predict = torch.randn(1, 1, 320, 320)
    alpha = predict
    # alpha = torch.randn(1, 1, 320, 320)

    # predict = torch.zeros(1, 1, 320, 320)
    # alpha = torch.ones(1, 1, 320, 320)
    
    # 实例化 LapLoss 类
    Lap = LapLoss().to(predict.device)

    # 计算损失大小
    loss = Lap(predict, alpha)
    print("{:.6f}".format(loss.item()))