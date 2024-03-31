import torch
def print_cuda():
    print('------------------- Pytorch InFo -------------------')
    print("PyTorch version: ", torch.__version__)
    print("CUDA: ", torch.cuda.is_available())
    if(torch.cuda.is_available()):
        print("CUDA version: ", torch.version.cuda)
        print("CUDA device count: ", torch.cuda.device_count())
        print("CUDA current device: ", torch.cuda.current_device())
    print('----------------------------------------------------')
