from functools import partial
import torch.nn as nn
import yaml

from model.Inception_model import Inception
from model.GoogLeNet_model import GoogLeNet

from model.vit import ViT
from model.mobile_vit import MobileViT
from model.vit_seg_modeling import VisionTransformer
from model.vit_seg_modeling import CONFIGS


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
    ALL_parameters['VisionTransformer_parameters'] = load_model_parameters('benchmark/config/model_VisionTransformer_parameters.yaml')

def load_VisionTransformer_config():
    use_QAT = ALL_parameters['VisionTransformer_parameters']['use_QAT']
    vit_name = ALL_parameters['VisionTransformer_parameters']['vit_name']
    img_size = ALL_parameters['VisionTransformer_parameters']['image_size'][0]
    vit_patches_size = ALL_parameters['VisionTransformer_parameters']['vit_patches_size']
    n_classes = ALL_parameters['VisionTransformer_parameters']['n_classes']
    n_skip = ALL_parameters['VisionTransformer_parameters']['n_skip']
    config_vit = CONFIGS[vit_name]
    config_vit.n_classes =n_classes
    config_vit.n_skip = n_skip
    config_vit.use_QAT = use_QAT
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    return config_vit, img_size

"""========== ALL Model =========="""
load_all_model_config_before_access()
config_vit, img_size = load_VisionTransformer_config()
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
    "VisionTransformer": VisionTransformer(
            config_vit, 
            img_size=img_size, 
            num_classes=config_vit.n_classes,
            use_QAT=config_vit.use_QAT
    )
}
