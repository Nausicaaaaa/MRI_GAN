import os
from net.pix2pix_HD_model.networks import define_D
from segmentation_models_pytorch import Unet
import safetensors.torch

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
encoder_weights_path = os.path.join(project_root, 'pretrained_weights', 'regnety_160.sw_in12k_ft_in1k', 'model.safetensors')

net_G = Unet(encoder_name='timm-regnety_160',
             encoder_weights=None,  # 不自动下载，使用本地权重
             encoder_depth=4,
             decoder_channels=[128, 64, 32, 16],
             decoder_use_batchnorm=True,
             in_channels=6, classes=1).to('cuda')

# 加载本地预训练权重
if os.path.exists(encoder_weights_path):
    print(f"Loading encoder weights from: {encoder_weights_path}")
    state_dict = safetensors.torch.load_file(encoder_weights_path)
    
    # 获取模型当前的状态字典，用于过滤不匹配的层
    model_state_dict = net_G.encoder.state_dict()
    
    # 过滤掉形状不匹配的层（通常是第一层卷积，因为输入通道数不同）
    filtered_state_dict = {}
    skipped_keys = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
            else:
                skipped_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in model")
    
    # 加载过滤后的权重
    missing_keys, unexpected_keys = net_G.encoder.load_state_dict(filtered_state_dict, strict=False)
    
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} layers due to shape mismatch:")
        for key in skipped_keys[:5]:
            print(f"  - {key}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    print(f"Encoder weights loaded successfully! ({len(filtered_state_dict)}/{len(state_dict)} layers loaded)")
else:
    print(f"Warning: Encoder weights not found at {encoder_weights_path}")
net_D = define_D(input_nc=7,
                 ndf=64,
                 n_layers_D=3,
                 norm="instance",
                 use_sigmoid=False,
                 num_D=2,
                 getIntermFeat=False)
print(net_D)
