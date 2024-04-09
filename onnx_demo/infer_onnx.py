
import cv2
from torchvision import transforms
import onnxruntime
import numpy as np

import sys
import os
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(config_path)
from config import *

# 定义Normalize函数
def normalize_image(image, mean, std):
    # image = image / 255.0  # 将图像像素值归一化到[0, 1]
    image = (image - mean) / std  # 根据均值和标准差进行归一化
    return image.astype(np.float32)

def preprocess_image(image, model_name):
    (w, h) = load_model_parameters(f'benchmark/config/model_{model_name}_parameters.yaml')['image_size']
    # Resize图像
    image = cv2.resize(image, (w, h))
    # 转换为Tensor并归一化
    image = normalize_image(image, [0.49372172, 0.46933405, 0.44654398], [0.30379174, 0.29378528, 0.30067085])
    image = np.transpose(image, (2, 0, 1))  # 调整维度顺序
    image = np.expand_dims(image, axis=0)  # 添加batch维度
    image = image.astype(np.float32)
    return image

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
        outputs = np.clip(outputs, 0, 1)
        outputs = (outputs * 255).astype(np.uint8) 
        cv2.imwrite(f"result/output_{i}.png", outputs)
        print(f"output_{i}.png saved.")


if __name__ == "__main__":

    model = "VisionTransformer"

    model_path = f"onnx_demo/{model}.onnx"
    image_path = "data/AIM500/test/original/p_f6b02429.jpg"

    # 加载图片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
    # 输入预处理
    image = preprocess_image(image, model)
    # 推理
    onnx_inference(model_path, image)