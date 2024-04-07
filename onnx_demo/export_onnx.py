import torch
import onnx
import torch.onnx

import sys
import os
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 1)[0]  # 上1级目录
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_CONFIG['MobileViT'].eval().to(device)
    checkpoint = torch.load(f'ckpt/MobileViT_53_pure.pkl', map_location=device)
    model.load_state_dict(checkpoint, False)
    converter = PyTorchToONNXConverter(model)
    input_shape = (1, 3, 512, 512)  # 示例输入大小:(None, 3, 256, 256) --> 模型输出大小:(None, 1, 256, 256)
    converter.convert_to_onnx(input_shape, 'onnx_demo/MobileViT.onnx', device)
    print("ONNX 模型导出完成！")

    # TODO: INT8量化

    # 检查导出的模型
    import onnx
    onnx_model = onnx.load('onnx_demo/MobileViT.onnx')
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    print("ONNX 模型检查通过！")