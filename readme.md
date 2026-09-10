# CE-MRI Synthesis: Contrast-Enhanced MRI from Multi-Sequence Non-Contrast MRI

Synthesize contrast-enhanced T1-weighted MRI (CE-MRI / T1+C) from multiple non-contrast MRI sequences using a Pix2PixHD-based GAN with an optional Transformer-based multi-sequence fusion module.

**Input sequences**: Pre-T1, T1, T2, DWI, ADC  
**Output**: Synthetic CE-MRI (T1+C)

## Model Architecture

```
                    ┌─────────────────────────────────┐
                    │   Multi-Sequence MRI Input       │
                    │   [Pre-T1, T1, T2, DWI, ADC]    │
                    │        Shape: [B, 5, H, W]      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Multi-Series Fusion Module     │
                    │                                  │
                    │  Option A: Concat (baseline)     │
                    │    → Direct channel passthrough  │
                    │                                  │
                    │  Option B: Transformer Fusion    │
                    │    → Spatial pooling             │
                    │    → Per-pixel modal attention   │
                    │    → Pixel-wise weighted output  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Generator (Pix2PixHD)          │
                    │                                  │
                    │  Encoder: RegNetY-160            │
                    │    (ImageNet pretrained)         │
                    │  Decoder: Upsampling + ResBlocks │
                    │  Output: [B, 1, H, W]            │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   Synthetic CE-MRI               │
                    └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │   Multi-Scale Discriminator      │
                    │   Input: [5 sequences + target]  │
                    │   Architecture: PatchGAN × 2     │
                    └─────────────────────────────────┘
```

### Transformer Fusion Module

The key innovation is a lightweight **Pixel-wise Modal Attention** module that learns spatially-adaptive sequence importance:

- Each pixel position gets independent sequence weights — e.g., tumor regions may attend more to DWI/ADC
- Spatial pooling → input projection → learnable modal type embedding → Transformer encoder (sequence length = 5) → output projection → bilinear upsample → pixel-wise weighting
- Only ~2,353 additional parameters
- Interpretable: attention weights can be visualized as heatmaps showing which sequence contributes most at each spatial location

### Training Losses

| Loss | Weight | Description |
|------|--------|-------------|
| LS-GAN | 8 | Least-squares adversarial loss |
| L1 | 25 | Voxel-wise reconstruction loss |
| SSIM | 20 | Structural similarity loss |

Optional: Perceptual loss (RadImageNet ResNet50), VGG feature matching loss, MS-SSIM loss.

## Project Structure

```
CE-MRI-synthesis/
├── net/
│   ├── multi_series_fusion.py          # Multi-sequence fusion (concat / transformer)
│   └── pix2pix_HD_model/               # Pix2PixHD generator & discriminator
│       ├── networks.py                 # Network definitions (GlobalGenerator, MultiscaleDiscriminator)
│       ├── pix2pixHD_model.py          # Model training logic
│       └── base_model.py
├── loss_function/
│   ├── losses_function.py              # Loss registry and definitions
│   ├── perceptual_loss.py              # Perceptual loss
│   ├── MS_SSIM.py                      # Multi-scale SSIM
│   └── ms_ssim_pytorch.py             # PyTorch MS-SSIM implementation
├── preprocessing/
│   ├── preprocessing.py                # Main preprocessing pipeline
│   ├── preprocessing_pipeline.py       # Pipeline orchestration
│   ├── preprocessing_nii/              # NIfTI processing (resampling, normalization, slicing)
│   ├── reg_series/                     # Image registration (ANTsPy-based)
│   └── resemble_n_get_body/            # Body mask extraction
├── training_project/
│   ├── train_main.py                   # Training entry point
│   ├── trainer_pix2pix_mulD.py         # Trainer with multi-discriminator
│   ├── ce_mri_param.py                 # Configuration (hyperparams, paths, loss weights)
│   ├── custom_transform.py             # Data loading & augmentation
│   └── utils/                          # Utilities (progress bar, visualization, etc.)
├── inference/
│   ├── inference_2d_main.py            # 2D inference pipeline
│   ├── visualize_attention_heatmap.py  # Attention weight visualization
│   ├── batch_attention_stats.py        # Batch attention statistics
│   └── postprocess_*.py               # Post-processing (artifact removal)
└── compare_results/
    └── visual_comparison/              # Prediction vs GT comparison tools
```

## Preprocessing Pipeline

```
DICOM → NIfTI → Resampling → Registration (ANTsPy) → Body Mask → Normalization → Empty Slice Removal → H5
```

1. **DICOM to NIfTI** conversion
2. **Resampling** to unified spatial resolution based on a standard spacing
3. **Registration** — all sequences aligned to T1 space using ANTsPy
4. **Body mask extraction** — threshold segmentation on T1 (intensity > 150) → hole filling → morphological operations, then applied to all sequences
5. **Normalization** and empty slice removal (effective region intersection across all 6 modalities)
6. **H5 conversion** for efficient training data loading

## Quick Start

### Requirements

```bash
pip install -r CE-MRI-synthesis/requirements.txt
```

Key dependencies: PyTorch, MONAI, ANTsPy, h5py, torchmetrics

### Training

```bash
cd CE-MRI-synthesis
python training_project/train_main.py
```

Key configurations in `training_project/ce_mri_param.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fusion_mode` | `concat` | Fusion mode: `concat` or `transformer` |
| `encoder_weights_path` | RegNetY-160 | Pretrained encoder path |
| `train_batch_size` | 4 | Batch size per GPU |
| `num_epochs` | 200 | Total training epochs |
| `lr` | 2e-5 | Initial learning rate |
| `transformer_embed_dim` | 24 | Transformer embedding dim |
| `transformer_spatial_pool_size` | 128 | Spatial pool size for attention |

### Inference

```bash
python inference/inference_2d_main.py
```

### Attention Visualization

```bash
python inference/visualize_attention_heatmap.py
```

## Results

### Quantitative Metrics

| Model | MS-SSIM ↑ | MI ↑ | NRMSE ↓ | SMAPE ↓ | LOGAC ↓ | MEDSYMAC ↓ |
|-------|-----------|------|---------|---------|---------|------------|
| GAN (concat) | 0.8853 | 1.1078 | 0.0648 | 0.0134 | 0.0268 | 0.0047 |
| GAN + Transformer | TBD | TBD | TBD | TBD | TBD | TBD |

### Concat vs Transformer Fusion

| | Concat (Baseline) | Transformer Fusion |
|---|---|---|
| Fusion strategy | Early fusion (channel concat) | Late fusion (pixel-wise attention) |
| Independent encoding | No | Yes (per-sequence CNN) |
| Cross-sequence interaction | No | Yes (self-attention) |
| Extra parameters | — | ~2.4K (attention module) |
| Total parameters | ~30M | ~30M + 2.4K |

## License

TBD
