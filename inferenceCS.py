import cv2
import os
import torch
from torchvision import transforms
from config import *
import numpy as np
import argparse

import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上1级目录
sys.path.append(config_path)
from config import *
(w, h) = load_model_parameters('benchmark/config/model_MobileViT_parameters.yaml')['image_size']

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载图片
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB

    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 转换为PIL图像
        transforms.Resize((w, h)),  # 与训练时相同的Resize大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.50542366, 0.46995255, 0.44692866], std=[0.28501507, 0.27542947, 0.28659645])  # 与训练时相同的Normalize参数
    ])

    # 对图片进行预处理
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # 添加batch维度

    # 加载模型
    model = MODEL_CONFIG['MobileViT']
    model.to(device)

    checkpoint = torch.load(f'ckpt/{args.model_name}.pkl', map_location=device)
    model.load_state_dict(checkpoint, False)
    model.eval()

    # 假设model是您加载的训练好的模型
    output = model(input_tensor)
    # 转为单通道8位的灰度图保存
    output = output.squeeze().cpu().detach().numpy()
    output = (output * 255).astype(np.uint8)  
    # 保存结果
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.image_path))[0] + '.png')
    cv2.imwrite(output_path, output)

    print(f"Inference completed. Output image saved as '{output_path}'.")

if __name__ == '__main__':
    # python inferenceCS.py --image_path data/AIM500/test/original/o_dc288b1a.jpg --model_name MobileViT_194_pure
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="data/AIM500/test/mask/o_dc288b1a.png", help='Path to the input image')
    parser.add_argument('--model_name', type=str, default="MobileViT_194_pure", help='Name of the model to use for inference')
    parser.add_argument('--output_dir', type=str, default='./result', help="Path to the output directory")
    args = parser.parse_args()

    main(args)