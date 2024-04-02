# EMA-ViTMatting

[[Project Page]](https://github.com/CSsaan/EMA-ViTMatting/) [[中文主页]](https://github.com/CSsaan/EMA-ViTMatting/README_CN.md)

Using EMA to train Matting task. 

Single RGB image input, single alpha image output.

This project focuses on the field of image alpha matting. Currently, there are few open-source end-to-end alpha matting models available, most of which are based on convolutional neural network models with large parameter sizes. Therefore, this paper adopts a mobile ViT combined with an improved cascaded decoder module to create a lightweight alpha matting model with reduced computational complexity. The innovation lies in the combination of a lightweight ViT model and an improved decoder module, bringing a more efficient solution to the alpha matting field.

## 👀 Demo

Demo: [Bilibili Video](https://www.bilibili.com/)

| **Original Image** | **Ground Truth Label** | **Training Results**  | **Test Results** |
| --- | --- | --- | --- |
| <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif"> |
| <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif"> |
| <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif">  | <img src="/.gif"> |

Model structure:
<img src="/.png" width="600">

## 📦 Prerequisites

#### Requirements:
- Python >= 3.8
- torch >= 2.2.2
- CUDA Version >= 11.7

## 🔧 Install

#### Configure Environment:

```bash
git clone git@github.com:CSsaan/EMA-ViTMatting.git
cd EMA-ViTMatting
conda create -n ViTMatting python=3.10 -y
conda activate ViTMatting
pip install -r requirements.txt
```

## 🚀 Quick Start

#### train script:
```bash
python train.py --use_model_name 'MobileViT' --reload_model False --local_rank 0 --world_size 4 --batch_size 16 --data_path 'data/classification/train' --use_distribute False
```
* `--use_model_name 'MobileViT'`: The name of the model to load
* `--reload_model False`: Model checkpoint continuation training
* `--local_rank 0`: The local rank of the current process
* `--world_size 4`: The total number of processes
* `--batch_size 16`: Batch size
* `--data_path 'data/classification/train'`: Data path
* `--use_distribute False`: Whether to use distributed training

#### test script:
```bash
python inferenceCS.py --image_path data/AIM500/test/original/o_dc288b1a.jpg --model_name MobileViT_194_pure
```

## 📖 Paper

None

## 🎯 Todo

- [x] Data preprocessing               -> dataset\AIM_500_datasets.py
- [x] Data augmentation                -> dataset\AIM_500_datasets.py
- [x] Model loading                    -> config.py & Trainer.py
- [x] Loss functions                   -> benchmark\loss.py
- [x] Dynamic learning rate            -> train.py
- [x] Distributed training             -> train.py
- [x] Model visualization              -> model\mobile_vit.py
- [x] Model parameters                 -> benchmark\config\model_MobileViT_parameters.yaml
- [x] Training                         -> train.py
- [x] Model saving                     -> Trainer.py
- [x] Test visualization               ->
- [x] Model inference                  -> inferenceCS.py
- [ ] Model acceleration               ->
- [ ] Model optimization               ->
- [ ] Model tuning                     ->
- [ ] Model integration                ->
- [ ] Model quantization, compression, deployment  ->

## 📂 Repo structure (WIP)

```
├── README.md
├── benchmark
│   ├── loss.py              -> loss functions
│   └── config               -> all model's parameters
├── utils
│   ├── testGPU.py
│   ├── yuv_frame_io.py
│   └── print_structure.py
├── data                     -> dataset
├── dataset                  -> dataloder
├── log                      -> tensorboard log
├── model
├── Trainer.py               -> load model & train.
├── config.py                -> all models dictionary
├── dataset.py               -> dataLoader
├── demo_Fc.py               -> model inder
├── pyproject.toml           -> project config
├── requirements.txt
├── train.py                 -> main
└── inferenceCS.py           -> model inference
```