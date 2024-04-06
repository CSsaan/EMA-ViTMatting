import os
import math
import time
import torch
import random
import argparse	
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from config import *
from Trainer import LoadModel
from dataset import load_classification_data
from dataset import AIM500Dataset
from benchmark.utils.testGPU import print_cuda

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000
        # return 2e-4 * mul
    else:
        mul = np.cos((step - 2000) / (300 * args.step_per_epoch - 2000) * 10 * np.pi) * 0.5 + 0.5
    return (1e-4 - 1e-5) * mul + 1e-5

def train(model, reloadModel_epochs, local_rank, batch_size, world_size, data_path):
    if local_rank == 0:
        writer = SummaryWriter('log/train_EMAVFI')
    step_train, step_eval, best = 0, 0, 0

    # --------- AIM-500 数据集加载 -----------------
    dataset = AIM500Dataset('train', root_dir=os.getcwd()+data_path)
    if(args.use_distribute):
        print('DataLoader use distribute.')
        sampler = DistributedSampler(dataset)
        train_data = DataLoader(dataset, batch_size=batch_size, num_workers=world_size, pin_memory=True, drop_last=True, shuffle=True, sampler=sampler)
    else:
        train_data = DataLoader(dataset, batch_size=batch_size, num_workers=world_size, pin_memory=True, drop_last=True, shuffle=True)
    dataset_val = AIM500Dataset('test', root_dir=os.getcwd()+data_path)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=world_size, pin_memory=True, drop_last=True, shuffle=True)
    # -----------------------------------------------------
    print("train_data.__len__(), val_data.__len__():", dataset.__len__(), dataset_val.__len__())

    args.step_per_epoch = train_data.__len__()

    print('---------------- training... -----------------------')
    time_stamp = time.time()
    min_loss = 10000

    # 断点续练
    start_epoch = 0
    if(reloadModel_epochs[0]):
        start_epoch = reloadModel_epochs[1]
        print('加载 epoch {} successed!'.format(start_epoch))

    # training loop epoch
    for epoch in tqdm(range(start_epoch+1, 300), desc='Epoch'):
        if(args.use_distribute):
            sampler.set_epoch(epoch)
        train_loss, train_l1_loss, train_mse_loss, train_l1_sobel_loss, train_laplacian_loss, train_iou_loss, train_dice_loss, train_com_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_loss, val_acc, val_num = 0.0, 0.0, 0.0
        pbar_batch = tqdm(train_data, desc='Training')
        for i, (data, target) in enumerate(pbar_batch):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # data = data.unsqueeze(1) # torch.Size([8]) -> torch.Size([8, 1])
            # target = target.unsqueeze(1) # torch.Size([8]) -> torch.Size([8, 1])
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            learning_rate = get_learning_rate(step_train)
            _, loss_dict = model.update(data, target, epoch, i, batch_size, learning_rate, training=True)
            train_loss += loss_dict['loss_all']
            train_l1_loss += loss_dict['l1_loss']
            train_mse_loss += loss_dict['mse_loss']
            train_l1_sobel_loss += loss_dict['l1_sobel_loss']
            train_laplacian_loss += loss_dict['laplacian_loss']
            train_iou_loss += loss_dict['iou_loss']
            train_dice_loss += loss_dict['dice_loss']
            train_com_loss += loss_dict['com_loss']

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step_train % 50 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step_train)
                writer.add_scalar('train/train_loss_all', train_loss.item()/(i+1), step_train)
                writer.add_scalar('loss/train_l1_loss', train_l1_loss/(i+1), step_train)
                writer.add_scalar('loss/train_mse_loss', train_mse_loss/(i+1), step_train)
                writer.add_scalar('loss/train_l1_sobel_loss', train_l1_sobel_loss/(i+1), step_train)
                writer.add_scalar('loss/train_laplacian_loss', train_laplacian_loss/(i+1), step_train)
                writer.add_scalar('loss/train_iou_loss', train_iou_loss/(i+1), step_train)
                writer.add_scalar('loss/train_dice_loss', train_dice_loss/(i+1), step_train)
                writer.add_scalar('loss/train_com_loss', train_com_loss/(i+1), step_train)
            postfix = {
                '[epoch]': epoch,
                'progress': '{}/{}'.format(i, args.step_per_epoch),
                'time': 'train:{:.2f}+continental:{:.2f}'.format(train_time_interval, data_time_interval),
                'loss_all': '{:.4f}'.format(train_loss.item()/(i+1)),
            }
            pbar_batch.set_postfix(postfix)  
            # if local_rank == 0:
            #     print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, train_loss/i))
            step_train += 1
        
        i = 1
        if epoch % 3 == 0:
            evaluate(model, val_data, epoch, i, local_rank, batch_size)
            i = 0

        if(train_loss.item()/step_train < min_loss):
            model.save_model(epoch, local_rank)
            min_loss = train_loss.item()
        
        # 分布式训练进程同步
        if(args.use_distribute):
            dist.barrier()

def evaluate(model, val_data, epoch, i, local_rank, batch_size):
    if local_rank == 0:
        writer_val = SummaryWriter('log/validate_EMA-Matting')

    loss = 1000
    for _, imgs in enumerate(val_data):
        data = imgs[0] # torch.Size([8]) -> torch.Size([8, 1])
        target = imgs[1] # torch.Size([8]) -> torch.Size([8, 1])
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.no_grad():
            pred, loss_dict = model.update(data, target, epoch, i, batch_size, training=False)
            loss = loss_dict['loss_all'].item()
        # for j in range(gt.shape[0]):
        #     loss.append(-10 * math.log10(((gt[j] - pred[j]) * (gt[j] - pred[j])).mean().cpu().item()))
   
    
    if local_rank == 0:
        print("*"*10, "test_loss", "*"*10)
        print(str(epoch), loss)
        writer_val.add_scalar('train/test_loss_all', loss, epoch)
        print("*"*30)
        
if __name__ == "__main__":    
    print_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_model_name', default='MobileViT', type=str, help='name of model to use') # 'GoogLeNet'、 'ViT'、 'MobileViT'
    parser.add_argument('--reload_model', default=True, type=bool, help='reload model')
    parser.add_argument('--reload_model_name', default='MobileViT_50', type=str, help='name of reload model')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=8, type=int, help='world size')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--data_path', default="/data/AIM500", type=str, help='data path of AIM_500 dataset')
    parser.add_argument('--use_distribute', default=False, type=bool, help='train on distribute Devices by torch.distributed')
    args = parser.parse_args()

    # 分布式训练
    if(args.use_distribute):
        torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
        torch.distributed.init_process_group(backend="gloo", world_size=args.world_size, timeout=datetime.timedelta(days=1))

    # 当前GPU索引
    if(torch.cuda.is_available()):
        torch.cuda.set_device(args.local_rank)
    # GPU0设备时，创建tensorboard的log路径
    if args.local_rank == 0 and not os.path.exists('log'):
        os.mkdir('log')
    
    # 设置随机种子
    set_random_seed(seed=1234, deterministic=False)

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = os.path.abspath('.').split('/')[-1]

    # 实例化模型
    model = LoadModel(args.local_rank, args.use_model_name, args.use_distribute)
    
    # 断电续练
    epochs = 0
    if (args.reload_model):
        epochs = model.reload_model(args.reload_model_name) # 继续训练加载的模型名字
    
    # 开始训练
    train(model, [args.reload_model, epochs], args.local_rank, args.batch_size, args.world_size, args.data_path)
        
