#!/usr/bin/env python3
"""
批量计算全部测试集的模态注意力权重统计（仅输出数字，不生成图片）

用法:
    source MRI_GAN/bin/activate
    cd CE-MRI-synthesis
    python -m inference.batch_attention_stats \
        --model_dir ../results/CE_MRI_simulate_PCa_1_trans_fold5-1-2 \
        --test_data_dir ../train-data/images_ts \
        --output_file ../output_attention_vis/all_patients_stats.json \
        --cuda 7
"""

import os
import sys
import glob
import json
import argparse
import time
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.test_param import config
from training_project.trainer_pix2pix_mulD import Pix2Pix_2d_MulD


def find_checkpoint(model_dir):
    """查找 checkpoint.ckpt"""
    ckpt = os.path.join(model_dir, "checkpoint", "checkpoint.ckpt")
    if os.path.exists(ckpt):
        return ckpt
    # fallback: 直接找 .ckpt
    ckpts = glob.glob(os.path.join(model_dir, "**/*.ckpt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"未找到 checkpoint: {model_dir}")
    # 优先选 epoch 最大的
    import re
    best, best_epoch = ckpts[0], -1
    for c in ckpts:
        m = re.search(r"epoch=(\d+)", c)
        if m and int(m.group(1)) > best_epoch:
            best_epoch = int(m.group(1))
            best = c
    return best


def run_batch_stats(model_dir, test_data_dir, output_file, cuda_idx=0, batch_size=16):
    """批量计算全测试集注意力统计"""
    train_keys = ["pret1", "t1", "t2", "dwi", "adc"]

    # ---- 加载模型 ----
    ckpt_path = find_checkpoint(model_dir)
    print(f"[Info] Checkpoint: {ckpt_path}")
    device = torch.device(f"cuda:{cuda_idx}")
    model = Pix2Pix_2d_MulD.load_from_checkpoint(
        ckpt_path,
        map_location=lambda storage, loc: storage.cuda(cuda_idx),
        weights_only=False
    )
    model.eval()
    print(f"[Info] Model loaded. fusion_mode={model.fusion_mode}")

    if model.fusion_mode != "transformer":
        print("[Warning] fusion_mode != 'transformer', 注意力权重可能不可用!")

    # ---- 扫描患者 ----
    patient_dirs = sorted([
        d for d in os.listdir(test_data_dir)
        if os.path.isdir(os.path.join(test_data_dir, d))
    ])
    print(f"[Info] Found {len(patient_dirs)} patients in {test_data_dir}")

    all_patient_stats = {}   # patient_id -> {modality_mean, modality_std, n_slices}
    global_sums = {k: 0.0 for k in train_keys}
    global_sq_sums = {k: 0.0 for k in train_keys}
    global_count = 0
    global_slice_means = {k: [] for k in train_keys}  # 收集每切片的均值

    t0 = time.time()

    for pi, pid in enumerate(patient_dirs):
        patient_dir = os.path.join(test_data_dir, pid)
        h5_files = sorted(glob.glob(os.path.join(patient_dir, "*.h5")))
        if not h5_files:
            continue

        # 批量加载和推理
        patient_slice_means = {k: [] for k in train_keys}

        for batch_start in range(0, len(h5_files), batch_size):
            batch_files = h5_files[batch_start:batch_start + batch_size]
            images = []

            for hf in batch_files:
                with h5py.File(hf, 'r') as f:
                    arrs = []
                    for k in train_keys:
                        a = f[k][()]
                        if a.ndim == 2:
                            a = np.expand_dims(a, axis=0)
                        arrs.append(a)
                    img = np.concatenate(arrs, axis=0)  # [5, H, W]
                    images.append(img)

            # 拼 batch [B, 5, H, W]
            batch_tensor = torch.from_numpy(np.stack(images, axis=0)).float().to(device)

            with torch.no_grad():
                _, attn = model(batch_tensor, return_attention=True)

            if attn is None:
                print(f"[Warning] attn is None for {pid}, skipping")
                continue

            # attn: [B, S, H, W]
            attn_np = attn.cpu().numpy()
            for bi in range(attn_np.shape[0]):
                for si, k in enumerate(train_keys):
                    m = float(attn_np[bi, si].mean())
                    patient_slice_means[k].append(m)
                    global_slice_means[k].append(m)
                    global_sums[k] += m
                    global_sq_sums[k] += m * m
                    global_count += 1

        # 患者级统计
        p_stats = {"n_slices": len(h5_files)}
        for k in train_keys:
            vals = patient_slice_means[k]
            if vals:
                p_stats[f"{k}_mean"] = float(np.mean(vals))
                p_stats[f"{k}_std"] = float(np.std(vals))
        all_patient_stats[pid] = p_stats

        elapsed = time.time() - t0
        print(f"  [{pi+1:3d}/{len(patient_dirs)}] {pid}: {len(h5_files)} slices, "
              f"pret1={p_stats.get('pret1_mean', 0):.3f}  "
              f"t1={p_stats.get('t1_mean', 0):.3f}  "
              f"t2={p_stats.get('t2_mean', 0):.3f}  "
              f"dwi={p_stats.get('dwi_mean', 0):.3f}  "
              f"adc={p_stats.get('adc_mean', 0):.3f}  "
              f"[{elapsed:.1f}s]")

    # ---- 全局统计 ----
    n_total = global_count // len(train_keys)  # 总切片数
    print(f"\n{'='*60}")
    print(f"Dataset-wide Statistics ({len(patient_dirs)} patients, {n_total} slices)")
    print(f"{'='*60}")

    dataset_means = {}
    dataset_stds = {}
    for k in train_keys:
        mean = global_sums[k] / n_total
        # std from E[x^2] - E[x]^2
        var = global_sq_sums[k] / n_total - mean * mean
        std = float(np.sqrt(max(var, 0)))
        dataset_means[k] = mean
        dataset_stds[k] = std
        print(f"  {k:8s}: mean={mean:.4f}  std={std:.4f}")

    # 排序
    ranked = sorted(train_keys, key=lambda k: dataset_means[k], reverse=True)
    print(f"\n  Ranking: {' > '.join(ranked)}")

    # ---- 保存结果 ----
    result = {
        "meta": {
            "model_dir": model_dir,
            "test_data_dir": test_data_dir,
            "n_patients": len(patient_dirs),
            "n_slices": n_total,
            "train_keys": train_keys,
        },
        "dataset_wide": {
            k: {"mean": dataset_means[k], "std": dataset_stds[k]}
            for k in train_keys
        },
        "ranking": ranked,
        "per_patient": all_patient_stats,
        "per_slice_means": {k: global_slice_means[k] for k in train_keys},
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Done] Results saved to: {output_file}")
    total_time = time.time() - t0
    print(f"[Done] Total time: {total_time:.1f}s ({total_time/len(patient_dirs):.1f}s per patient)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, default='../train-data/images_ts')
    parser.add_argument('--output_file', type=str, default='../output_attention_vis/all_patients_stats.json')
    parser.add_argument('--gpu', type=int, default=7, help='GPU index')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    run_batch_stats(args.model_dir, args.test_data_dir, args.output_file, args.gpu, args.batch_size)
