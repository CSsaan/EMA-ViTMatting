# EMA-ViTMatting
Using EMA to train Matting task


Python >= 3.8
torch >= 1.8.0
CUDA Version >= 11.7
skimage 0.19.2
numpy 1.23.1
opencv-python 4.6.0
timm 0.6.11
tqdm


git clone git@github.com:CSsaan/EMA-ViTMatting.git
cd EMA-ViTMatting
conda create -n EMA python=3.8 -y
conda activate EMA
pip install -r requirements.txt


## 📂 Repo structure (WIP)
```
├── README.md
├── Trainer.py                    -> load model & train.
├── config.py                     -> all models dictionary
├── dataset.py                    -> dataLoader
├── demo_Fc.py                    -> model inder
├── pyproject.toml                ->  project config
├── requirements.txt
├── train.py                      -> main
├── benchmark
│   ├── loss.py
│   └── config                    -> all model's parameters
└── utils
│   ├── testGPU.py
│   ├── yuv_frame_io.py
│   └── print_structure.py
├── data                          -> dataset
├── log                           -> tensorboard log
└── model
```
