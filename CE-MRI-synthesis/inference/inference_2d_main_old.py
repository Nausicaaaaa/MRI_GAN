import os.path
import re
import sys
import argparse
import glob

# 添加项目根目录到 sys.path，确保能正确导入 inference 和 training_project 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import SimpleITK as sitk
import pandas as pd
import lightning.pytorch as pl
import torch
from lightning.pytorch import seed_everything
from monai.utils import set_determinism

from inference.test_param import config
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD
from inference.test_metrics import nrmse, smape, logac, medsymac, scale12bit
from inference.get_metric import compute_ms_ssim, compute_mi

torch.multiprocessing.set_sharing_strategy('file_system')
set_determinism(2023)
seed_everything(2023, workers=True)

if __name__ == "__main__":
    # torch.set_float32_matmul_precision('high')
    # ==========path============
    Task_name = config.Task_name
    task_id = config.Task_id
    fold_idx = config.fold_idx
    ckpt_name = config.ckpt_name
    # 使用相对于脚本的路径计算项目根目录
    dir_prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config.result_path = os.path.join(dir_prefix, config.result_path)
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.h5_2d_img_dir = os.path.join(dir_prefix, 'train-data')
    # ===============model setting==============
    task_name = "{}_{}_{}_fold5-{}".format(Task_name, task_id, config.net_mode, fold_idx)
    result_path = config.result_path
    # ================search for best==============================
    ckpt_dir = os.path.join(result_path, task_name, "checkpoint")
    print(f"[Inference] Checkpoint dir: {ckpt_dir}")
    ckpt_list = os.listdir(ckpt_dir)
    pattern = r"{}(-epoch=\d+)?\.ckpt".format(ckpt_name.split('.')[0])
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
    versions = [re.search(r"epoch=(\d+)", file).group(1) for file in ckpt_file if re.search(r"epoch=\d+", file)]
    sorted_versions = sorted(versions, key=lambda x: int(x))
    ckpt_to_resume = f"{ckpt_name.split('.')[0]}-epoch={sorted_versions[-1]}.ckpt" if sorted_versions else ckpt_name

    ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
    print(f"[Inference] Resume from: {ckpt_to_resume}")
    # 根据指定的GPU列表设置map_location，自动将权重映射到目标设备
    target_device = f"cuda:{config.cuda_idx_list[0]}"
    model = Pix2Pix_2d_MulD.load_from_checkpoint(ckpt_path,
                                                 map_location={"cuda:0": target_device},
                                                 weights_only=False
                                                 )
    # 覆盖模型中的路径属性，确保即使项目被移动也能正确找到数据
    model.template_dir = config.filepath_img
    model.data_dir = config.h5_2d_img_dir
    model.test_dir = os.path.join(model.data_dir, "images_ts")
    # 将输出明确放到 results 目录下
    model.pred_result_dir = os.path.join(result_path, task_name,
                                         f"pred_nii_{model.fusion_mode}_{ckpt_name.split('.')[0]}")
    if not os.path.exists(model.pred_result_dir):
        os.makedirs(model.pred_result_dir)
    print(f"[Inference] Output dir: {model.pred_result_dir}")
    # ==========PL MODEL============
    torch.set_float32_matmul_precision('high')
    print("========================{}==========================".format(task_name))
    # 推理使用单GPU，避免DDP数据分片导致预测结果不完整
    # batch_size=1时多GPU不会加速，反而造成rank0只写入部分结果
    infer_device = [config.cuda_idx_list[0]]
    print(f"[Inference] Using GPU: {infer_device}")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=infer_device,
        enable_progress_bar=False,
        # limit_predict_batches=2
    )
    predictions = trainer.predict(model)
