import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# netron模型可视化
from torchsummary import summary
import netron
import torch.onnx
from torch.autograd import Variable

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # 正方形的image、patch块
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # image的宽高可以整除patch宽高（正好分割成多个patch）
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        # patch数量
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # [1, 3, 256, 256]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # [1, 3, (8*32) (8*32)] -> [1, (8*8), (32*32*3)] . 转换成大小：(b, patch数, 每个patch的3通道像素)
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), # [1, 64, (32*32*3)] -> [1, 64, 1024]. 转换：将每个patch的3通道像素进行全连接，大小：(b, patch数, 每个patch的3通道像素的全连接)
            nn.LayerNorm(dim),
            # [1, 64, 1024]
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # [1, 1, 1024]
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img) # [1, 3, 256, 256] -> [1, 64, 1024] 将所有patch的小图转为全连接大小，结果大小(b, patch数, 每个patch的3通道像素的全连接)
        b, n, _ = x.shape # 大小：(b, patch数, 全连接输出大小)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # [1, 1, 1024] -> [b, 1, 1024]  一个patch
        x = torch.cat((cls_tokens, x), dim=1) # [1, 65, 1024] 加一个patch
        x += self.pos_embedding[:, :(n + 1)]  # [1, 65, 1024]
        x = self.dropout(x)

        x = self.transformer(x) # [1, 65, 1024]
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0] # [1, 65, 1024] -> [1, 1024]

        x = self.to_latent(x)

        return self.mlp_head(x) # [1, 1024] -> [1, 1000]


if __name__ == "__main__":
    # 打印模型网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(
        image_size = 256, # 原图输入大小
        patch_size = 32, # 每个patch大下
        num_classes = 1000,
        dim = 1024, # 每个patch全连接后大小
        depth = 2, # Transformer个数
        heads = 16, # 注意力头个数
        mlp_dim = 2048, # 全连接大小
        pool = 'mean', # {'cls', 'mean'}
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    print(summary(model, (3, 224, 224)))

    # 可视化模型网络结构
    input_x = torch.randn(8, 3, 224, 224)  # 随机生成一个输入
    modelData = "./demo.pth"  # 定义模型数据保存的路径
    # modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
    torch.onnx.export(model, input_x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
    netron.start(modelData)  # 输出网络结构