import torch
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pyr_downsample(x):
    return x[:, :, ::2, ::2]


def pyr_upsample(x, kernel, op0, op1):
    n_channels, _, kw, kh = kernel.shape
    return F.conv_transpose2d(x, kernel, groups=n_channels, stride=2, padding=2, output_padding=(op0, op1))

def gauss_kernel5(channels=3, cuda=True):
    kernel = torch.FloatTensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    # print(kernel)
    if cuda:
        kernel = kernel.cuda()
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


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels-1):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current) # high -> low
    return pyr

def laplacian_pyramid_expand(img, kernel, max_levels=5):
    current = img
    pyr = []
    for level in range(max_levels):
        # print("level: ", level)
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        up = pyr_upsample(down, 4*kernel, 1-filtered.size(2)%2, 1-filtered.size(3)%2)

        diff = current - up
        pyr.append(diff)

        current = down
    return pyr


class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self._gauss_kernel = None

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = gauss_kernel5(input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid_expand(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
        weights = [1, 2, 4, 8, 16]

        # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean() 


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
        Lap = LapLoss().to(predict.device)
        Laposs = Lap(predict, alpha)
        l1 = F.l1_loss(predict, alpha)
        l1_sobel = F.l1_loss(kornia.filters.sobel(predict), kornia.filters.sobel(alpha))
        mse_loss = F.mse_loss(predict, alpha)
        
        return Laposs + l1 + 0.5*l1_sobel + mse_loss # + laplacian_loss

if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    predict = torch.randn(1, 1, 256, 256)
    alpha = predict
    # alpha = torch.randn(1, 1, 256, 256)

    # predict = torch.zeros(1, 1, 256, 256)
    # alpha = torch.ones(1, 1, 256, 256)

    # 实例化 MattingLoss 类
    matting_loss = MattingLoss().to(predict.device)

    # 计算损失大小
    loss = matting_loss(predict, alpha)
    print("{:.6f}".format(loss.item()))