import torch
import os
import cv2
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from benchmark.loss import *
from torchsummary import summary
from config import *

    
class LoadModel:
    def __init__(self, local_rank, model_name='GoogLeNet', use_distribute=False, use_QAT=False):
        if(model_name =='GoogLeNet'):
            inception_model, googlelenet_model = MODEL_CONFIG[model_name]
            self.net = googlelenet_model(inception_model)
        elif(model_name =='ViT' or model_name =='MobileViT' or model_name =='VisionTransformer'):
            self.net = MODEL_CONFIG[model_name]

        self.name = model_name
        self.use_QAT = use_QAT
        print(f'loaded model: {self.name}')
        
        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.loss = MattingLoss()
        if (use_distribute and local_rank != -1):
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)
        # move model and loss function to device
        self.device()

        # 量化感知训练
        if (self.use_QAT):
            print(f'use QAT')
            self.prepare_QAT()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(_device)
        self.loss.to(_device)
        print(f'Model use Device: {_device}')

    def prepare_QAT(self):
        # 模型必须设置为 eval 才能使融合工作
        self.eval()
        # 附加一个全局的qconfig，其中包含关于要附加哪种观察器的信息。对于服务器推断，请使用'x86'，对于移动推断，请使用'qnnpack'。
        # 其他量化配置，如选择对称或非对称量化以及MinMax或L2Norm校准技术，可以在这里指定。
        # 注意：旧的'fbgemm'仍然可用，但'x86'是服务器推断的推荐默认值。
        # self.net.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        self.net.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        # 将激活与前面的层融合，如果适用的话，这需要根据模型架构手动完成
        # self.net = torch.ao.quantization.fuse_modules(self.net, [['transformer', 'decoder', 'segmentation_head', 'to_logits']])
        # 为量化感知训练准备模型。这会在模型中插入观察器和伪量化器，
        # 模型需要设置为训练以使量化感知逻辑生效，该模型将在校准期间观察权重和激活张量。
        self.net = torch.ao.quantization.prepare_qat(self.net.train())

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
    
    def save_model(self, epoch, arg, rank=0):
        if rank == 0:
            self.net.eval()
            if (self.use_QAT):
                self.net = torch.ao.quantization.convert(self.net)
            # 仅保存模型
            save_dir = 'ckpt'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # 保存模型参数(纯模型结构，不支持断点续训)
            torch.save(self.net.state_dict(), os.path.join(save_dir, f'{self.name}_{arg}_{epoch}_pure.pkl')) # {str(epoch)}
            # 支持断点续训
            state = {'model': self.net.state_dict(), 'optimizer': self.optimG.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join('ckpt', f'{self.name}_{arg}_{epoch}.pkl')) # {str(epoch)}

    @torch.no_grad()
    def inference(self, input):
        preds = self.net(input)
        return preds
    
    def update(self, inputX, imputY, epoch, i, batch_size, learning_rate=0.001, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()

        if training:
            pred = self.net(inputX)
            _loss_ = (self.loss(pred.unsqueeze(1), imputY, inputX)) # b w h -> b c w h

            # 保存每个batch结果图
            if(epoch % 1 == 0 and i == 1):
                final_image = None
                for i in range(batch_size):
                    first_batch_rgb = inputX[i].permute(1, 2, 0).cpu().detach().numpy()
                    first_batch_rgb = cv2.cvtColor(first_batch_rgb, cv2.COLOR_RGB2BGR)
                    # output
                    batch_list = torch.split(pred, 1, dim=0)
                    selected_image = batch_list[i].squeeze().cpu().detach().numpy()
                    selected_image_rgb = np.stack((selected_image, selected_image, selected_image), axis=-1)
                    # inputY
                    batch_imputY = torch.split(imputY, 1, dim=0)
                    selected_imageY = batch_imputY[i].squeeze().cpu().detach().numpy()
                    selected_imageY_rgb = np.stack((selected_imageY, selected_imageY, selected_imageY), axis=-1)
                    # concatenate
                    concatenated_image = np.concatenate((first_batch_rgb, selected_imageY_rgb, selected_image_rgb), axis=0)
                    if final_image is None:
                        final_image = concatenated_image
                    else:
                        final_image = np.concatenate((final_image, concatenated_image), axis=1)
                cv2.imwrite('result/batch_result_train.png', final_image*255)

            self.optimG.zero_grad()
            _loss_['loss_all'].backward()
            self.optimG.step()
            return pred, _loss_
        else: 
            with torch.no_grad():
                pred = self.net(inputX)
                _loss_ = (self.loss(pred.unsqueeze(1), imputY, inputX))

                # 保存每个batch结果图
                if(i == 1):
                    final_image = None
                    for i in range(batch_size):
                        first_batch_rgb = inputX[i].permute(1, 2, 0).cpu().detach().numpy()
                        first_batch_rgb = cv2.cvtColor(first_batch_rgb, cv2.COLOR_RGB2BGR)
                        # output
                        batch_list = torch.split(pred, 1, dim=0)
                        selected_image = batch_list[i].squeeze().cpu().detach().numpy()
                        selected_image_rgb = np.stack((selected_image, selected_image, selected_image), axis=-1)
                        # inputY
                        batch_imputY = torch.split(imputY, 1, dim=0)
                        selected_imageY = batch_imputY[i].squeeze().cpu().detach().numpy()
                        selected_imageY_rgb = np.stack((selected_imageY, selected_imageY, selected_imageY), axis=-1)
                        # concatenate
                        concatenated_image = np.concatenate((first_batch_rgb, selected_imageY_rgb, selected_image_rgb), axis=0)
                        if final_image is None:
                            final_image = concatenated_image
                        else:
                            final_image = np.concatenate((final_image, concatenated_image), axis=1)
                    cv2.imwrite('result/batch_result_test.png', final_image*255)

                return pred, _loss_


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = LoadModel(local_rank=0, use_distribute=False)
    model = load_model.net.to(device)
    print(summary(model, (3, 224, 224)))