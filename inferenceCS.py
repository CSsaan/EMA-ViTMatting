import cv2
import os
import torch
from config import *
import numpy as np
import argparse

import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上1级目录
sys.path.append(config_path)
from config import *

def normalize_image(image, mean, std):
    image = image / 255.0  # 将图像像素值归一化到[0, 1]
    # image = (image - mean) / std  # 根据均值和标准差进行归一化
    return image.astype(np.float32)

def preprocess_image(image, model_name):
    (w, h) = load_model_parameters(f'benchmark/config/model_{model_name}_parameters.yaml')['image_size']
    # Resize图像
    image = cv2.resize(image, (w, h))
    # 转换为Tensor并归一化
    # image = normalize_image(image, [0.49372172, 0.46933405, 0.44654398], [0.30379174, 0.29378528, 0.30067085])
    image = np.transpose(image, (2, 0, 1))  # 调整维度顺序
    image = np.expand_dims(image, axis=0)  # 添加batch维度
    return image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载图片
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB

    # 输入预处理
    image = preprocess_image(image, args.model)
    # 转tensor
    input_tensor = torch.from_numpy(image).to(device)
    input_tensor = input_tensor.to(torch.float32)
    print(input_tensor.size())
    # 加载模型
    model = MODEL_CONFIG[args.model]
    model.to(device)

    checkpoint = torch.load(f'ckpt/{args.model_name}.pkl', map_location=device)
    model.load_state_dict(checkpoint, False)
    model.eval()

    # 假设model是您加载的训练好的模型
    output = model(input_tensor)
    # 转为单通道8位的灰度图保存
    output = output.squeeze().cpu().detach().numpy()
    # clamp到0-1之间
    output = np.clip(output, 0, 1)
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
    parser.add_argument('--image_path', type=str, default="/workspaces/EMA-ViTMatting/data/AIM500/train/original/o_1b4c1dfc.jpg", help='Path to the input image')
    parser.add_argument('--model', type=str, default="VisionTransformer", help='Name of the model to use for inference')
    parser.add_argument('--model_name', type=str, default="VisionTransformer_64_pure", help='Name of the model state_dict')
    parser.add_argument('--output_dir', type=str, default='./result', help="Path to the output directory")
    args = parser.parse_args()

    main(args)