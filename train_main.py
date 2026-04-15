# -*- coding: utf-8 -*-

import os
import re
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CE-MRI-synthesis'))

import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from monai.utils import set_determinism

from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD
from training_project.ce_mri_param import config
import argparse

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_determinism(config.seed)
    seed_everything(config.seed, workers=True)
    # 设置好路径 - 使用项目根目录
    dir_prefix = os.path.dirname(os.path.abspath(__file__))
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.h5_3d_img_dir = os.path.join(dir_prefix, config.h5_3d_img_dir)
    config.h5_2d_img_dir = os.path.join(dir_prefix, config.h5_2d_img_dir)
    config.filepath_mask = os.path.join(dir_prefix, config.filepath_mask)
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # 设置 encoder 权重路径
    if not os.path.isabs(config.encoder_weights_path):
        config.encoder_weights_path = os.path.join(dir_prefix, config.encoder_weights_path)
    # 设置任务名和对应的路径
    # CE_MRI_simulate_1_2d_fold5-1
    task_name = config.Task_name + '_' + config.Task_id + '_' + config.net_mode + '_fold' + str(
        config.fold_K) + "-" + str(
        config.fold_idx)
    print("===================={}=====================".format(task_name))
    root_dir = os.path.join(config.result_path, task_name)
    config.root_dir = os.path.join(config.result_path, task_name)
    # config.record_file = os.path.join(config.root_dir, "log_txt.txt")
    # ===============================set up GPU======================================================
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    # 设置多GPU环境变量
    if len(config.cuda_idx_list) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.cuda_idx_list))
        print(f"Using GPUs: {config.cuda_idx_list}")
    # =====================================set up loggers and checkpoints======================================================
    log_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
    # tensorboard --logdir = log_dir
    # ===================================callback======================================================
    ckpt_dir = os.path.join(root_dir, "checkpoint")
    loss_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="val_loss_best",
        monitor='val_loss',
        mode="min",
        save_last=False,
        save_top_k=1,
        save_weights_only=True,
    )
    best_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch}",
        monitor='val_ssim',
        mode="max",
        save_last=False,
        save_top_k=2,
        save_weights_only=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="checkpoint",
        every_n_epochs=config.checkpoint_epoch,
        save_on_train_epoch_end=True
    )

    # =================================initialise Lightning's trainer.======================================================
    profiler = AdvancedProfiler(dirpath=root_dir, filename="perf_logs")
    
    # 根据GPU数量设置devices参数
    if len(config.cuda_idx_list) > 1:
        devices_config = config.cuda_idx_list
        strategy_config = "ddp"  # 分布式数据并行
    else:
        devices_config = [config.cuda_idx]
        strategy_config = "auto"
    
    trainer = pl.Trainer(
        # default_root_dir=root_dir,
        accelerator='gpu',
        devices=devices_config,
        strategy=strategy_config,
        max_epochs=config.num_epochs,
        check_val_every_n_epoch=config.val_step,
        logger=tb_logger,
        enable_checkpointing=True,
        log_every_n_steps=1,
        callbacks=[best_callback, checkpoint_callback],
        deterministic="warn",
        enable_progress_bar=False,
        # =====dev option=====
        num_sanity_val_steps=0,
        # fast_dev_run=1,
        # limit_train_batches=1,
        limit_val_batches=4,
        # limit_train_batches=300,
        # profiler=profiler,
    )
    # ===================configurate net===================================
    unet = Pix2Pix_2d_MulD(config)
    # ========================search for ckpt==============================
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_list = os.listdir(ckpt_dir)
    pattern = r"checkpoint(-v\d+)?\.ckpt"
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
    # 新训练
    if not ckpt_file:
        print("========== No checkpoint to resume, start a new train ==========")
        trainer.fit(unet)
    # 断点恢复
    else:
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"checkpoint-v{sorted_versions[-1]}.ckpt" if sorted_versions else "checkpoint.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)

        trainer.fit(unet, ckpt_path=ckpt_path)
