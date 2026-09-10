# -*- coding: utf-8 -*-

import os
import re
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

# PyTorch 2.6+ compatibility: default weights_only changed to True, which breaks loading
# checkpoints containing argparse.Namespace and other objects. We patch torch.load to
# default weights_only=False for compatibility with existing checkpoints.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # PyTorch 2.6 treats weights_only=None as True internally, so we must override it explicitly
    if 'weights_only' not in kwargs or kwargs.get('weights_only') is None:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

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
    # 设置好路径 - 使用项目根目录 (train_main.py 在 CE-MRI-synthesis/training_project/ 下)
    dir_prefix = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config.filepath_img = os.path.join(dir_prefix, config.filepath_img)
    config.h5_3d_img_dir = os.path.join(dir_prefix, config.h5_3d_img_dir)
    config.h5_2d_img_dir = os.path.join(dir_prefix, config.h5_2d_img_dir)
    config.filepath_mask = os.path.join(dir_prefix, config.filepath_mask)
    config.result_path = os.path.join(dir_prefix, config.result_path)
    # 设置 encoder 权重路径
    if not os.path.isabs(config.encoder_weights_path):
        config.encoder_weights_path = os.path.join(dir_prefix, config.encoder_weights_path)
    # 设置 pretrained_ckpt 路径（相对路径时基于项目根目录解析）
    if config.pretrained_ckpt and not os.path.isabs(config.pretrained_ckpt):
        config.pretrained_ckpt = os.path.join(dir_prefix, config.pretrained_ckpt)
    # 设置任务名和对应的路径
    # CE_MRI_simulate_1_2d_fold5-1
    # 根据 fusion_mode 决定 net_mode 后缀
    net_mode_suffix = "trans" if config.fusion_mode == "transformer" else config.net_mode
    task_name = config.Task_name + '_' + config.Task_id + '_' + net_mode_suffix + '_fold' + str(
        config.fold_K) + "-" + str(
        config.fold_idx) + config.dir_suffix
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
        # 保存原始 GPU 编号
        original_gpu_list = config.cuda_idx_list
        # 设置环境变量，PyTorch 会重新映射 GPU 编号
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.cuda_idx_list))
        # Lightning 看到的 GPU 编号变为 0,1,2,3...
        devices_config = list(range(len(config.cuda_idx_list)))
        print(f"[Main] Physical GPUs {original_gpu_list} -> Logical GPUs {devices_config}")
    else:
        devices_config = [config.cuda_idx]
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
        # 已经在上一步设置了 devices_config
        strategy_config = "ddp_find_unused_parameters_true"  # Pix2Pix交替训练需允许未使用参数
        print(f"[Main] Using DDP strategy with GPUs: {devices_config}")
    else:
        devices_config = [config.cuda_idx]
        strategy_config = "auto"
        print(f"[Main] Using single GPU: {devices_config}")
    
    trainer = pl.Trainer(
        # default_root_dir=root_dir,
        accelerator='gpu',
        devices=devices_config,
        strategy=strategy_config,
        precision='32',  # 使用FP32保证训练稳定性
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
        # limit_val_batches=4,  # 修复：使用完整验证集进行评估
        # limit_train_batches=300,
        # profiler=profiler,
    )
    # ===================configurate net===================================
    print("[Main] Initializing Pix2Pix_2d_MulD model...")
    unet = Pix2Pix_2d_MulD(config)
    print("[Main] Model initialized successfully.")
    
    # 加载预训练模型（如果指定）
    if config.pretrained_ckpt and os.path.exists(config.pretrained_ckpt):
        print(f"[Main] Loading pretrained checkpoint from: {config.pretrained_ckpt}")
        # 先加载到 CPU，避免 GPU 内存问题
        checkpoint = torch.load(config.pretrained_ckpt, map_location='cpu', weights_only=False)
        # 处理不同格式的 checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉不匹配的层（例如不同融合模式的层）
        model_state_dict = unet.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        for key, value in state_dict.items():
            # 去掉 "module." 前缀（如果是 DDP 训练的）
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            
            if new_key in model_state_dict:
                if value.shape == model_state_dict[new_key].shape:
                    filtered_state_dict[new_key] = value
                else:
                    skipped_keys.append(f"{new_key}: shape mismatch {value.shape} vs {model_state_dict[new_key].shape}")
            else:
                skipped_keys.append(f"{new_key}: not in model")
        
        # 加载过滤后的权重
        missing_keys, unexpected_keys = unet.load_state_dict(filtered_state_dict, strict=False)
        
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} layers due to mismatch:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print(f"Pretrained weights loaded! ({len(filtered_state_dict)}/{len(state_dict)} layers loaded)")
        
        # 清理内存
        del checkpoint
        del state_dict
        del filtered_state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif config.pretrained_ckpt:
        print(f"[Main] Warning: pretrained checkpoint not found: {config.pretrained_ckpt}")
    
    # ========================search for ckpt==============================
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_list = os.listdir(ckpt_dir)

    # 优先从普通断点 checkpoint 恢复（包含优化器状态，可完整恢复训练）
    pattern = r"checkpoint(-v\d+)?\.ckpt"
    ckpt_file = [file for file in ckpt_list if re.match(pattern, file)]
    if ckpt_file:
        versions = [re.search(r"v(\d+)", file).group(1) for file in ckpt_file if re.search(r"v\d+", file)]
        sorted_versions = sorted(versions, key=lambda x: int(x))
        ckpt_to_resume = f"checkpoint-v{sorted_versions[-1]}.ckpt" if sorted_versions else "checkpoint.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
        print(f"[Main] Found regular checkpoint: {ckpt_path}")
        resume_with_ckpt_path = True
    else:
        # fallback: 从 best checkpoint 加载模型权重（仅权重，优化器重新初始化）
        best_pattern = r"best-epoch=(\d+)\.ckpt"
        best_ckpt_files = [f for f in ckpt_list if re.match(best_pattern, f)]
        if best_ckpt_files:
            epochs = [int(re.search(best_pattern, f).group(1)) for f in best_ckpt_files]
            best_epoch = max(epochs)
            ckpt_to_resume = f"best-epoch={best_epoch}.ckpt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_to_resume)
            print(f"[Main] Found BEST checkpoint: {ckpt_path} (epoch={best_epoch})")
            print("[Main] Warning: best checkpoint contains only model weights. Optimizer will be reinitialized.")
            resume_with_ckpt_path = False
        else:
            ckpt_path = None
            resume_with_ckpt_path = False

    # 新训练
    if ckpt_path is None:
        print("========== No checkpoint to resume, start a new train ==========")
        print("[Main] Starting trainer.fit()...")
        trainer.fit(unet)
        print("[Main] trainer.fit() completed.")
    # 断点恢复
    else:
        if resume_with_ckpt_path:
            print(f"[Main] Resuming from checkpoint: {ckpt_path}")
            print("[Main] Starting trainer.fit()...")
            trainer.fit(unet, ckpt_path=ckpt_path)
            print("[Main] trainer.fit() completed.")
        else:
            print(f"[Main] Loading weights from best checkpoint: {ckpt_path}")
            print("[Main] Starting trainer.fit()...")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            
            # 处理 spatial_conv 结构变更的兼容性（从单卷积改为 Bottleneck 结构）
            state_dict = checkpoint['state_dict']
            old_spatial_conv_key = "net_G.fusion_module.fusion_module.spatial_conv.weight"
            old_spatial_conv_bias = "net_G.fusion_module.fusion_module.spatial_conv.bias"
            
            if old_spatial_conv_key in state_dict:
                print("[Main] Detected old spatial_conv structure, migrating to new Bottleneck structure...")
                # 删除旧的空间卷积权重
                del state_dict[old_spatial_conv_key]
                if old_spatial_conv_bias in state_dict:
                    del state_dict[old_spatial_conv_bias]
                print("[Main] Removed old spatial_conv weights, new layers will be randomly initialized.")
                
                # 结构变更时，删除优化器状态（参数组已变化，不兼容）
                if 'optimizer_states' in checkpoint:
                    print("[Main] Removing optimizer states due to model structure change...")
                    del checkpoint['optimizer_states']
                if 'lr_schedulers' in checkpoint:
                    print("[Main] Removing lr_schedulers due to model structure change...")
                    del checkpoint['lr_schedulers']
            
            unet.load_state_dict(state_dict, strict=False)
            # 从文件名解析 epoch，手动恢复 epoch 计数
            best_epoch = int(re.search(r"best-epoch=(\d+)", ckpt_to_resume).group(1))
            print(f"[Main] Manually restoring epoch count to {best_epoch}...")
            
            # 清空 trainer 的 ckpt_path，防止 Lightning 尝试恢复优化器状态
            trainer.ckpt_path = None
            
            # 同步设置 fit_loop 的各项进度，使训练从 best_epoch 之后继续
            for tracker_name in ("current", "total"):
                tracker = getattr(trainer.fit_loop.epoch_progress, tracker_name)
                tracker.completed = best_epoch
                tracker.ready = best_epoch
                tracker.started = best_epoch
                tracker.processed = best_epoch
            trainer.fit(unet)
            print("[Main] trainer.fit() completed.")
