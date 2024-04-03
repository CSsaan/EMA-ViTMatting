from functools import partial
import torch.nn as nn
import yaml

from model.Inception_model import Inception
from model.GoogLeNet_model import GoogLeNet
from model.mobile_vit import MobileViT

from model.vit import ViT

def load_model_parameters(file_path):
    with open(file_path, 'r') as file:
        model_params = yaml.safe_load(file)
    return model_params


"""========== load Model config =========="""
ALL_parameters = {
    "ViT_parameters": None,
    "MobileViT_parameters": None,
}

def load_all_model_config_before_access():
    global ALL_parameters
    ALL_parameters['ViT_parameters'] = load_model_parameters('benchmark/config/model_ViT_parameters.yaml')
    ALL_parameters['MobileViT_parameters'] = load_model_parameters('benchmark/config/model_MobileViT_parameters.yaml')

"""========== ALL Model =========="""
load_all_model_config_before_access()
MODEL_CONFIG = {
    "LOGNAME": "CS_ALL_MODEL",
    "GoogLeNet": (Inception, GoogLeNet),
    "ViT": ViT(
            image_size = ALL_parameters['ViT_parameters']['image_size'], # 原图输入大小
            patch_size = ALL_parameters['ViT_parameters']['patch_size'], # 每个patch大下
            num_classes = ALL_parameters['ViT_parameters']['num_classes'],
            dim = ALL_parameters['ViT_parameters']['dim'], # 每个patch全连接后大小
            depth = ALL_parameters['ViT_parameters']['depth'], # Transformer个数
            heads = ALL_parameters['ViT_parameters']['heads'], # 注意力头个数
            mlp_dim = ALL_parameters['ViT_parameters']['mlp_dim'], # 全连接大小
            pool = ALL_parameters['ViT_parameters']['pool'], # {'cls', 'mean'}
            dropout = ALL_parameters['ViT_parameters']['dropout'],
            emb_dropout = ALL_parameters['ViT_parameters']['emb_dropout']
            ),
    "MobileViT": MobileViT(
            image_size = ALL_parameters['MobileViT_parameters']['image_size'], # 原图输入大小
            dims = ALL_parameters['MobileViT_parameters']['dims'],
            channels = ALL_parameters['MobileViT_parameters']['channels'],
            depths = ALL_parameters['MobileViT_parameters']['depths'],
            use_cat = ALL_parameters['MobileViT_parameters']['use_cat']
            ),
}
