# 深度学习模型合成卵巢MR增强序列

## 数据

**输入**：非增强序列(ADC、AX_T1、AX_T2、DWI、AX_preT1)
**输出**：合成的对比增强序列 AX_T1 +/1+

|A中心|B中心|C中心|
|--|--|--|
|训练集/测试集|外部验证集|外部验证集|
|730例|--|--|
|80G|--|--|


## 代码结构

```
MRI_GAN/
├── CE-MRI-synthesis/           # 核心代码目录
│   ├── inference/              # 推理相关代码
│   │   ├── get_metric.py       # 评估指标计算
│   │   ├── inference_2d_main.py # 2D推理主程序
│   │   ├── test_metrics.py     # 测试指标
│   │   └── test_param.py       # 测试参数
│   ├── loss_function/          # 损失函数
│   │   ├── MS_SSIM.py          # MS-SSIM损失
│   │   ├── losses_function.py  # 自定义损失函数
│   │   ├── ms_ssim_pytorch.py  # PyTorch版MS-SSIM
│   │   └── perceptual_loss.py  # 感知损失
│   ├── net/                    # 网络模型
│   │   ├── pix2pix_HD_model/   # Pix2PixHD模型
│   │   │   ├── base_model.py
│   │   │   ├── models.py
│   │   │   ├── networks.py
│   │   │   ├── pix2pixHD_model.py
│   │   │   └── ui_model.py
│   │   └── multi_series_fusion.py # 多序列融合模块
│   ├── preprocessing_nii/      # 数据预处理
│   │   ├── kill_empty_slice.py # 去除空切片
│   │   ├── normalization_2_new_ni_pca.py # 归一化
│   │   ├── preprocess_h5_2d.py # 预处理为h5格式
│   │   └── resize_nii.py       # 图像重采样
│   ├── reg_series/             # 图像配准
│   │   ├── reg_param/          # 配准参数
│   │   ├── antspy_registration.py
│   │   ├── itk_resample_all2t1.py
│   │   ├── preprocess.py
│   │   └── resample_t1.py
│   ├── resemble_n_get_body/    # 身体区域提取
│   │   ├── get_body_all_series.py
│   │   └── get_body_t1.py
│   └── training_project/       # 训练相关代码
│       ├── utils/              # 工具函数
│       ├── ce_mri_param.py     # 参数配置
│       ├── train_main.py       # 训练主程序
│       └── trainer_pix2pix_mulD.py # 训练器
├── data/                       # 原始DICOM数据
│   └── {患者ID}/
│       ├── ADC/
│       ├── AX_PreT1/
│       ├── AX_T1/
│       ├── AX_T1 1+/
│       ├── AX_T2/
│       └── DWI/
└── pre-data/                   # 预处理后的数据
    └── body_masks/             # 身体掩膜
```

## 环境
激活环境：`source MRI_GAN/bin/activate`

# 项目流程
## 1. 数据预处理

`run_preprocessing.py`

1. DICOM转NIfTI
2. 序列重采样为统一尺寸（基于一个标准尺寸和space）
3. 基于T1序列配准对齐（关键：ANTsPy）
4. 基于T1序列获取身体掩码 （get_body_t1.py：阈值分割（img_array > 150） → 孔洞填充 → 形态学操作 ）
5. 将身体掩码应用至其他序列（可选）
6. 归一化+去除不包含有效信息的切片
7. 转换为h5格式


## 2. 模型训练

training_project/ce_mri_param.py 配置训练参数(设置任务名称、GPU、输入序列、批次大小、学习率、损失权重等 )
training_project/trainer_pix2pix_mulD.py
training_project/custom_transform.py 数据加载和预处理
train_main.py 启动训练(使用 Pix2PixHD + RegNet 架构进行训练 生成器: RegNet 骨干网络 判别器: 多尺度判别器)

开始训练：
```python train_main.py```
|阶段|步骤|concat模式	|transformer模式|
|--|--|--|--|
|数据预处理| H5 加载 |从 H5 加载 5 个序列 + t1ce|相同|
||通道处理 |ConcatKeys 拼接为 [B,5,H,W] / t1ce 为 [B,1,H,W]|相同|
||数据增强|随机旋转、翻转等|相同|
|模型输入|real_A (输入)	|[B, 5, H, W]	|[B, 5, H, W]|
||real_B (目标)|	[B, 1, H, W]|	[B, 1, H, W]
|多序列融合	|融合模块|	MultiSeriesFusionWrapper (无操作，直接返回)|	MultiSeriesTransformerFusion|
||处理方式	|早期融合，通道拼接后直接进入 UNet	|晚期融合，先通过 Transformer 学习序列关系|
||独立编码	|❌ 无|	✅ 每个序列独立 CNN 编码|
||序列间交互	|❌ 无	|✅ 自注意力机制学习序列关系|
||空间下采样	|❌ 无	|✅ 下采样到 H/4 × W/4|
||位置编码	|❌ 无	|✅ 2D 正弦位置编码|
||输出形状	|[B, 5, H, W] |[B, 5, H, W] |
|生成器| 输入通道| 5|	5|
||架构|	UNet (RegNetY-160 编码器)|	相同|
||输出	|[B, 1, H, W] (生成 CE-MRI)	|相同|
|判别器|	输入	|[B, 6, H, W] |(5序列+1目标/生成)	|相同|
||架构|	多尺度 PatchGAN	|相同|
|损失计算|	损失函数|	GAN + L1 + SSIM|	相同|
||反向传播	|标准流程|	相同|
|配置参数|	额外参数|	无	|transformer_embed_dim (默认 256)|
||||transformer_num_heads (默认 8)|
||||transformer_num_layers (默认 4)|
||||transformer_dropout (默认 0.1)|
||||transformer_spatial_size (默认 320)|
|资源消耗	|显存占用|	较低|	较高 (额外 CNN + Transformer)|
||训练速度	|快	|慢 (约 2-3 倍时间)|
|参数量	||少 (~30M)	|多 (+~15M Transformer 参数)|
|适用场景	||	快速迭代、基线实验	|需要建模复杂序列关系、提升精度|
  

## 3. 结果分析
### 3.1 定量评估

模拟图与真实图、不同DL模型间对比

**a、相似性指标**

- 多尺度结构相似性指数（MS-SSIM）
- 直方图互信息（MI）

 **b、误差指标**

- 归一化均方根误差（NRMSE）
- 对称平均绝对百分比误差（SMAPE）
- 对数准确率比（LOGAC）
- 中位数对称准确率（MEDSYMAC）

### 3.2 定性评估

 **图像质量的评分**
 
 三名放射科医生盲法评估图像质量，基于以下标准使用5分Likert量表评估(1 = nondiagnostic because of impaired image quality; 2 = poor; 3 = acceptable; 4 = good; and 5 = excellent)：分两次评估，间隔一个月，每次评估两个系列，真实和模拟图混合，同一患者的真实图和模拟图不能在同一个系列中。参考标准：图像对比度、图像清晰度、视觉信噪比

**病变大小（最大径）测量对比**

**临床应用评估**

对多发病变，选取最大者或形态结构最复杂者分析

a、使用O-RADS评分（模拟与真实图分别评估，并对比差异）
b、诊断良恶性的准确率、敏感性、特异性、F1值，评分≥4分定为恶性