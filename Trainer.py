import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from benchmark.loss import *
from torchsummary import summary
from config import *
import os
    
class LoadModel:
    def __init__(self, local_rank, model_name='GoogLeNet', use_distribute=False, n_channels=1):
        if(model_name =='GoogLeNet'):
            inception_model, googlelenet_model = MODEL_CONFIG[model_name]
            self.net = googlelenet_model(inception_model)
        elif(model_name =='ViT'):
            self.net = MODEL_CONFIG[model_name]
        elif(model_name =='MobileViT'):
            self.net = MODEL_CONFIG[model_name]

        self.name = model_name
        print(f'loaded model: {self.name}')
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.loss = MattingLoss(n_channels=n_channels)
        if (use_distribute and local_rank != -1):
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(_device)
        print(f'Model use Device: {_device}')

    def reload_model(self, name=None):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if name is None:
            raise Exception("reload model name is None!")
        # 加载保存的模型，来断点续练
        checkpoint = torch.load(f'ckpt/{name}.pkl')
        self.net.load_state_dict(checkpoint['model'], False)
        self.optimG.load_state_dict(checkpoint['optimizer'])
        epochs = checkpoint['epoch']
        return epochs
        

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if name is None:
                name = self.name
            # self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl')), False)
            checkpoint = torch.load(f'ckpt/{name}.pkl')
            self.net.load_state_dict(checkpoint['model'], False)
    
    def save_model(self, epoch, rank=0):
        if rank == 0:
            # 仅保存模型
            save_dir = 'ckpt'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数
            torch.save(self.net.state_dict(), f'{save_dir}/{self.name}_{str(epoch)}_pure.pkl')
            # 支持断点续练
            state = {'model': self.net.state_dict(), 'optimizer': self.optimG.state_dict(), 'epoch': epoch}
            torch.save(state, f'ckpt/{self.name}_{str(epoch)}.pkl')

    @torch.no_grad()
    def inference(self, input):
        preds = self.net(input)
        return preds
    
    def update(self, inputX, imputY, learning_rate=0.001, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            pred = self.net(inputX)
            loss_mse = (self.loss(pred, imputY))
            # loss_mse = F.mse_loss(pred, imputY)

            # BUG: 保存结果
            import cv2
            batch_list = torch.split(pred, 1, dim=0)
            selected_image = batch_list[0].squeeze()
            print(selected_image.size())
            cv2.imwrite('selected_image_opencv.png', selected_image.detach().numpy()*255)

            self.optimG.zero_grad()
            loss_mse.backward()
            self.optimG.step()
            return pred, loss_mse
        else: 
            with torch.no_grad():
                pred = self.net(inputX)
                loss_mse = (self.loss(pred, imputY))
                return pred, loss_mse


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = LoadModel(local_rank=0, use_distribute=False)
    model = load_model.net.to(device)
    print(summary(model, (3, 224, 224)))