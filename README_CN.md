# EMA-ViTMatting

使用 EMA 训练抠图任务。

单通道 RGB 图像输入，单通道 alpha 图像输出。

这个项目专注于图像 alpha 抠图领域。目前，可用的开源端到端 alpha 抠图模型很少，其中大多基于参数规模较大的卷积神经网络模型。因此，本文采用了移动 ViT 结合改进的级联解码器模块，创建了一个轻量级 alpha 抠图模型，降低了计算复杂度。创新之处在于轻量级 ViT 模型和改进的解码器模块的结合，为 alpha 抠图领域带来了更高效的解决方案。

## 👀 演示

演示：[Bilibili 视频](https://www.bilibili.com/)

| **原图** | **标签** | **训练结果**  | **测试结果** | --- | **原图** | **标签** | **训练结果**  | **测试结果** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <img src="result/p_f7b2317f.jpg">  | <img src="result/lab_p_f7b2317f.png">  | <img src="result/pre_p_f7b2317f.png">  | <img src="result/green_p_f7b2317f.png"> | --- | <img src="result/p_f89c7881.jpg">  | <img src="result/lab_p_f89c7881.png">  | <img src="result/pre_p_f89c7881.png">  | <img src="result/green_p_f89c7881.png"> |
| <img src="result/p_f30f22fd.jpg">  | <img src="result/lab_p_f30f22fd.png">  | <img src="result/pre_p_f30f22fd.png">  | <img src="result/green_p_f30f22fd.png"> | --- | <img src="result/p_fcb9a19e.jpg">  | <img src="result/lab_p_fcb9a19e.png">  | <img src="result/pre_p_fcb9a19e.png">  | <img src="result/green_p_fcb9a19e.png"> |
| <img src="result/p_f053bec5.jpg">  | <img src="result/lab_p_f053bec5.png">  | <img src="result/pre_p_f053bec5.png">  | <img src="result/green_p_f053bec5.png"> | --- | <img src="result/p_fe6a4bfe.jpg">  | <img src="result/lab_p_fe6a4bfe.png">  | <img src="result/pre_p_fe6a4bfe.png">  | <img src="result/green_p_fe6a4bfe.png"> |
| <img src="result/p_f879fac6.jpg">  | <img src="result/lab_p_f879fac6.png">  | <img src="result/pre_p_f879fac6.png">  | <img src="result/green_p_f879fac6.png"> | --- | <img src="result/p_fdaa48dd.jpg">  | <img src="result/lab_p_fdaa48dd.png">  | <img src="result/pre_p_fdaa48dd.png">  | <img src="result/green_p_fdaa48dd.png"> |


模型结构:
<img src="/.png" width="600">

## 📦 先决条件

#### 环境要求:
- Python >= 3.8
- torch >= 2.2.2
- CUDA 版本 >= 11.7

## 🔧 安装

#### 配置环境:

```bash
git clone git@github.com:CSsaan/EMA-ViTMatting.git
cd EMA-ViTMatting
conda create -n ViTMatting python=3.10 -y
conda activate ViTMatting
pip install -r requirements.txt
```

## 🚀 快速开始

#### 训练脚本:

数据集目录结构：
data
└── AIM500
    ├── train
    │   ├── original
    │   └── mask
    └── test
        ├── original
        └── mask
```

```bash
python train.py --use_model_name 'VisionTransformer' --reload_model False --local_rank 0 --world_size 4 --batch_size 16 --data_path '/data/AIM500' --use_distribute False
```

#### 测试脚本:
```bash
python inferenceCS.py --image_path data/AIM500/test/original/o_dc288b1a.jpg --model_name MobileViT_194_pure
```

## 📖 论文

无

## 🎯 待办事项

- [x] 数据预处理               -> dataset\AIM_500_datasets.py
- [x] 数据增强                 -> dataset\AIM_500_datasets.py
- [x] 模型加载                 -> config.py & Trainer.py
- [x] 损失函数                 -> benchmark\loss.py
- [x] 动态学习率               -> train.py
- [x] 分布式训练               -> train.py
- [x] 模型可视化               -> model\mobile_vit.py
- [x] 模型参数                 -> benchmark\config\model_MobileViT_parameters.yaml
- [x] 训练                     -> train.py
- [x] 模型保存                 -> Trainer.py
- [x] 测试可视化               -> 
- [x] 模型推理                 -> inferenceCS.py
- [x] pytorch模型转onnx        -> onnx_demo
- [ ] 模型加速                 ->
- [ ] 模型优化                 ->
- [ ] 模型调参                 ->
- [ ] 模型集成                 ->
- [ ] 模型量化、压缩、部署    ->

## 📂 仓库结构 (WIP)

```

├── README.md
├── benchmark
│   ├── loss.py              -> 损失函数
│   └── config               -> 所有模型参数
├── utils
│   ├── testGPU.py
│   ├── yuv_frame_io.py
│   └── print_structure.py
├── onnx_demo
│   ├── export_onnx.py
│   └── infer_onnx.py
├── data                     -> 数据集
├── dataset                  -> 数据加载器
├── log                      -> tensorboard 日志
├── model
├── Trainer.py               -> 加载模型 & 训练.
├── config.py                -> 所有模型字典
├── dataset.py               -> 数据加载器
├── demo_Fc.py               -> 模型索引
├── pyproject.toml           -> 项目配置
├── requirements.txt
├── train.py                 -> 主程序
└── inferenceCS.py           -> 模型推理
```