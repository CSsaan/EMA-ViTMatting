
import cv2
import torch
from torchvision import transforms
import onnxruntime
import numpy as np

import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上1级目录
sys.path.append(config_path)
from config import *
(w, h) = load_model_parameters('benchmark/config/model_MobileViT_parameters.yaml')['image_size']

def onnx_inference(model_path, input):
    # 使用onnxruntime-gpu在GPU上进行推理
    session = onnxruntime.InferenceSession(model_path,
        providers=[
            ("CUDAExecutionProvider", {  # 使用GPU推理
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                # "cudnn_conv_use_max_workspace": "1"    # 在初始化阶段需要占用好几G的显存
            }),
            "CPUExecutionProvider"       # 使用CPU推理
        ])

    # session = onnxruntime.InferenceSession(model_path)

    # 获取模型原始输入的字段名称
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 以字典方式将数据输入到模型中
    outputs = session.run([output_name], {input_name: input}) # 模型输出：(None, 1, 256, 256)

    # 根据模型输出的batch大小，保存每个batch的灰度图
    for i in range(outputs[0].shape[0]):
        outputs = outputs[0]
        outputs = outputs.squeeze()
        outputs = (outputs * 255).astype(np.uint8) 
        cv2.imwrite(f"result/output_{i}.png", outputs)


if __name__ == "__main__":
    model_path = "onnx_demo/MobileViT.onnx"
    image_path = "data/AIM500/test/mask/o_dc288b1a.png"

    # 加载图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
    # 定义预处理操作
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 转换为PIL图像
        transforms.Resize((w, h)),  # 与训练时相同的Resize大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.50542366, 0.46995255, 0.44692866], std=[0.28501507, 0.27542947, 0.28659645])  # 与训练时相同的Normalize参数
    ])
    # 对图片进行预处理
    input_tensor = preprocess(image).unsqueeze(0)  # 添加batch维度
    # tensor转为numpy
    input = input_tensor.numpy()

    # input = np.random.randn(2, 3, 224, 224).astype(np.float32)
    onnx_inference(model_path, input)