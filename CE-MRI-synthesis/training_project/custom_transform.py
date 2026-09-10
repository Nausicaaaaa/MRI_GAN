# -*- coding: utf-8 -*-
"""
数据加载和预处理Transform 主要功能：
1. LoadH5 - 从 H5 文件加载数据（多序列输入 + t1ce 目标）
2. ConcatKeys - 将多个输入序列（pret1, t1, t2, dwi, adc）在通道维度拼接成 real_A
3. EnsureChannelFirst - 确保 t1ce 有通道维度
"""

import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, RandRotated, RandFlipd, RandScaleIntensityd,
    ToTensord, Resized, MapTransform
)
from training_project.utils.my_transform import LoadH5


class ConcatKeys(MapTransform):
    """将多个序列键拼接成一个image键"""
    def __init__(self, keys, output_key="image"):
        super().__init__(keys)
        self.output_key = output_key
    
    def __call__(self, data):
        d = dict(data)
        # 将多个序列在通道维度拼接 [C, H, W]
        arrays = []
        for key in self.keys:
            arr = d[key]
            # 确保每个数组至少是2维的，如果是2D则添加通道维度
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0)
            arrays.append(arr)
        d[self.output_key] = np.concatenate(arrays, axis=0)
        return d


class EnsureChannelFirst(MapTransform):
    """确保指定键的数据有通道维度 [1, H, W]"""
    def __init__(self, keys):
        super().__init__(keys)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key]
            # 如果是2D数组 [H, W]，添加通道维度变为 [1, H, W]
            if arr.ndim == 2:
                d[key] = np.expand_dims(arr, axis=0)
        return d


def get_2d_train_transform(keys, random_prob=0.4):
    """
    训练集数据预处理（已增强：随机翻转、旋转、强度缩放）

    Args:
        keys: H5文件中的键名列表，如 ["pret1", "t1", "t2", "dwi", "adc"]
        random_prob: 数据增强概率
    """
    # 同时加载 mask，用于空间对齐和后续 mask-weighted loss
    all_keys = keys + ["t1ce", "mask"]
    # 空间变换插值模式：图像序列用 bilinear，mask 用 nearest
    mode_list = ["bilinear"] * (len(all_keys) - 1) + ["nearest"]
    return Compose([
        LoadH5(path_key="path", keys=all_keys),
        # 先给所有数据补上通道维度 [1, H, W]，避免 MONAI 空间变换对纯2D数据产生维度歧义
        EnsureChannelFirst(keys=all_keys),
        # 数据增强：随机左右翻转（沿 W 轴）
        RandFlipd(keys=all_keys, spatial_axis=1, prob=random_prob),
        # 数据增强：随机上下翻转（沿 H 轴）
        RandFlipd(keys=all_keys, spatial_axis=0, prob=random_prob),
        # 数据增强：随机旋转 ±15°
        RandRotated(keys=all_keys, range_x=0.26, prob=random_prob, mode=mode_list),
        # 数据增强：随机强度缩放（仅输入序列）
        RandScaleIntensityd(keys=keys, factors=0.1, prob=random_prob),
        ConcatKeys(keys=keys, output_key="image"),  # 拼接输入序列 [5, H, W]
        ToTensord(keys=["image", "t1ce", "mask"]),  # 转换为Tensor
    ])


def get_2d_val_transform(keys):
    """
    验证集数据预处理（同时加载 mask，便于后续 mask-weighted 评估）

    Args:
        keys: H5文件中的键名列表
    """
    return Compose([
        LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
        EnsureChannelFirst(keys=keys + ["t1ce", "mask"]),
        ConcatKeys(keys=keys, output_key="image"),
        ToTensord(keys=["image", "t1ce", "mask"]),
    ])


def get_2d_test_transform(keys):
    """
    测试集数据预处理

    Args:
        keys: H5文件中的键名列表
    """
    return Compose([
        LoadH5(path_key="path", keys=keys + ["t1ce", "mask"]),
        EnsureChannelFirst(keys=keys + ["t1ce", "mask"]),
        ConcatKeys(keys=keys, output_key="image"),
        ToTensord(keys=["image", "t1ce", "mask"]),
    ])
