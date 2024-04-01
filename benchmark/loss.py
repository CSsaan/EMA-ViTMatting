import torch
import numpy as np 
import torch.nn.functional as F
from torch.autograd import Variable
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)

def laplacian_pyramid(img, kernel, max_levels=5):
	current = img
	pyr = []
	for level in range(max_levels):
		filtered = conv_gauss(current, kernel)
		diff = current - filtered
		pyr.append(diff)
		current = F.avg_pool2d(filtered, 2)
	pyr.append(current)
	return pyr

# class LapLoss(torch.nn.Module):
#     def __init__(self, max_levels=5, channels=3):
#         super(LapLoss, self).__init__()
#         self.max_levels = max_levels
#         self.gauss_kernel = gauss_kernel(channels=channels)
        
#     def forward(self, input, target):
#         pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
#         pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
#         return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

# class MyLoss(torch.nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#         self.loss_name = 'MSE'
#         self.loss = nn.CrossEntropyLoss()

#     def forward(self, input, target):
#         return self.loss(input, target)


# class TVLoss(torch.nn.Module):
#     def __init__(self, tv_loss_weight=1):
#         super(TVLoss, self).__init__()

# class Ternary(torch.nn.Module):
#     def __init__(self, device):
#         super(Ternary, self).__init__()
#         patch_size = 7
#         out_channels = patch_size * patch_size
#         self.w = np.eye(out_channels).reshape(
#             (patch_size, patch_size, 1, out_channels))
#         self.w = np.transpose(self.w, (3, 2, 0, 1))
#         self.w = torch.tensor(self.w).float().to(device)

#     def transform(self, img):
#         patches = F.conv2d(img, self.w, padding=3, bias=None)
#         transf = patches - img
#         transf_norm = transf / torch.sqrt(0.81 + transf**2)
#         return transf_norm

#     def rgb2gray(self, rgb):
#         r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
#         gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#         return gray

#     def hamming(self, t1, t2):
#         dist = (t1 - t2) ** 2
#         dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
#         return dist_norm

#     def valid_mask(self, t, padding):
#         n, _, h, w = t.size()
#         inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
#         mask = F.pad(inner, [padding] * 4)
#         return mask

#     def forward(self, img0, img1):
#         img0 = self.transform(self.rgb2gray(img0))
#         img1 = self.transform(self.rgb2gray(img1))
#         return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class MattingLoss(torch.nn.Module):
    def __init__(self, n_channels, cuda=False):
        super(MattingLoss, self).__init__()
        self.cuda = cuda
        self.n_channels = n_channels

    ## Laplacian loss is refer to 
    ## https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    def build_gauss_kernel(self, size=5, sigma=1.0, n_channels=1):
        if size % 2 != 1:
            raise ValueError("kernel size must be uneven")
        grid = np.float32(np.mgrid[0:size,0:size].T)
        gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        kernel = np.tile(kernel, (n_channels, 1, 1))
        kernel = torch.FloatTensor(kernel[:, None, :, :]).cuda() if self.cuda else torch.FloatTensor(kernel[:, None, :, :])
        return Variable(kernel, requires_grad=False)

    def mse_loss(self, predict, alpha):
        weighted = torch.ones(alpha.shape).cuda() if self.cuda else torch.ones(alpha.shape)
        alpha_f = alpha
        alpha_f = alpha_f.cuda() if self.cuda else alpha_f
        diff = predict - alpha_f
        alpha_loss = torch.sqrt(diff ** 2 + 1e-12)
        alpha_loss = alpha_loss.sum() / (weighted.sum())
        return alpha_loss

    def forward(self, predict, alpha):
        _mse = self.mse_loss(predict, alpha)
        l1 = F.l1_loss(predict, alpha)
        l1_sobel = F.l1_loss(kornia.filters.sobel(predict), kornia.filters.sobel(alpha))
        mse_loss = F.mse_loss(predict, alpha)

        # gauss_kernel = self.build_gauss_kernel(size=5, sigma=1.0, n_channels=self.n_channels)
        # pyr_alpha  = laplacian_pyramid(alpha_f, gauss_kernel, 5)
        # pyr_predict = laplacian_pyramid(predict, gauss_kernel, 5)
        # laplacian_loss = sum(F.l1_loss(a, b) for a, b in zip(pyr_alpha, pyr_predict))
        
        return l1 + l1_sobel + mse_loss # + laplacian_loss

if __name__ == '__main__':
    # 假设有两张图像 predict 和 alpha，均为 torch.Tensor 类型
    predict = torch.randn(1, 1, 256, 256)
    # alpha = predict
    alpha = torch.randn(1, 1, 256, 256)

    # predict = torch.zeros(1, 1, 256, 256)
    # alpha = torch.ones(1, 1, 256, 256)

    # 实例化 MattingLoss 类
    matting_loss = MattingLoss(n_channels=3, cuda=False)

    # 计算损失大小
    loss = matting_loss(predict, alpha)
    print("{:.6f}".format(loss.item()))