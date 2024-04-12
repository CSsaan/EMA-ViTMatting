import torch
import onnx
import torch.onnx
from PIL import Image
from torchvision import transforms

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers
import numpy as np


import sys
import os
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(config_path)
from config import *
		
class PyTorchToONNXConverter:
    def __init__(self, model):
        self.model = model

    def convert_to_onnx(self, input_shape, output_path, device):
        # 创建虚拟输入
        dummy_input = torch.randn(*input_shape).to(device)
        # 导出 ONNX 模型
        torch.onnx.export(
            self.model, # pytorch网络模型
            dummy_input, # 随机的模拟输入
            output_path, # 导出的onnx文件位置
            export_params=True, # 导出训练好的模型参数
            verbose=False, # debug message
            input_names=['input'], # 为静态网络图中的输入节点设置别名，在进行onnx推理时，将input_names字段与输入数据绑定
            output_names=['output'], # 为输出节点设置别名
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
        })


# 数据预处理
val_transforms = transforms.Compose(
    [
        # Resize(256, interpolation="bilinear"),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((224, 224)),
        # CenterCrop(224),
        transforms.Normalize((0.49372172, 0.46933405, 0.44654398), (0.30379174, 0.29378528, 0.30067085)),
    ]
)
 
# 数据批次读取器
def batch_reader(datas, batch_size):
    _datas = []
    length = len(datas)
    for i, data in enumerate(datas):
        if batch_size==1:
            yield {'input': data}
        elif (i+1) % batch_size==0:
            _datas.append(data)
            yield {'input': _datas}
            _datas = []
        elif i<length-1:
            _datas.append(data)
        else:
            _datas.append(data)
            yield {'input': _datas}
 
# 构建校准数据读取器
'''
    实质是一个迭代器
    get_next 方法返回一个如下样式的字典
    {
        输入 1: 数据 1, 
        ...
        输入 n: 数据 n
    }
    记录了模型的各个输入和其对应的经过预处理后的数据
'''
class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size):
        self.datas = batch_reader(datas, batch_size)
 
    def get_next(self):
        return next(self.datas, None)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_CONFIG['VisionTransformer'].eval().to(device)
    checkpoint = torch.load(f'ckpt/VisionTransformer_157_pure.pkl', map_location=device)
    model.load_state_dict(checkpoint, False)
    converter = PyTorchToONNXConverter(model)
    input_shape = (1, 3, 224, 224)  # 示例输入大小:(None, 3, 256, 256) --> 模型输出大小:(None, 1, 256, 256)
    converter.convert_to_onnx(input_shape, 'onnx_demo/VisionTransformer.onnx', device)
    print("ONNX 模型导出完成！")

    # 动态量化
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model_fp32 = 'onnx_demo/VisionTransformer.onnx'
    model_quant_dynamic = 'onnx_demo/VisionTransformer_quant_dynamic.onnx'
    quantize_dynamic(
        model_input=model_fp32, # 输入模型
        model_output=model_quant_dynamic, # 输出模型
        weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
    )
    print("ONNX 动态量化完成！")

    # TODO: 静态量化
    model_quant_static = 'onnx_demo/VisionTransformer_quant_static.onnx'
    img_dir = '/workspaces/EMA-ViTMatting/data/AIM500/train/original'
    img_num = 4
    datas = [
        val_transforms(
            Image.open(os.path.join(img_dir, img)).convert('RGB')
        ) for img in os.listdir(img_dir)[:img_num]
    ]
    datas = [data.numpy() for data in datas] # 将datas列表的每个元素转换为numpy类型
    data_reader = DataReader(datas, batch_size=2) # 实例化一个校准数据读取器
    quantize_static(
        model_input=model_fp32, # 输入模型
        model_output=model_quant_static, # 输出模型
        calibration_data_reader=data_reader, # 校准数据读取器
        quant_format= QuantFormat.QDQ, # 量化格式 QDQ / QOperator
        activation_type=QuantType.QInt8, # 激活类型 Int8 / UInt8
        weight_type=QuantType.QInt8, # 参数类型 Int8 / UInt8
        calibrate_method=CalibrationMethod.MinMax, # 数据校准方法 MinMax / Entropy / Percentile
    )
    print("ONNX 静态量化完成！")

    # 检查导出的模型
    import onnx
    onnx_model = onnx.load('onnx_demo/VisionTransformer.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    print("ONNX 模型检查通过！")