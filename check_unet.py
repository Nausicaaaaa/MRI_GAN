from segmentation_models_pytorch import Unet
import torch

base_generator = Unet(
    encoder_name='timm-regnety_160',
    encoder_weights=None,
    encoder_depth=4,
    decoder_channels=[128, 64, 32, 16],
    decoder_use_batchnorm=True,
    in_channels=5,
    classes=1
)

# 检查最后一层
for name, module in base_generator.named_children():
    print(f'{name}: {type(module).__name__}')

# 直接测试输出
x = torch.randn(1, 5, 256, 256)
with torch.no_grad():
    out = base_generator(x)
print(f'Output shape: {out.shape}')
print(f'Output min: {out.min():.4f}, max: {out.max():.4f}, mean: {out.mean():.4f}, std: {out.std():.4f}')

# 检查是否最后一层有激活函数
print("\nLast few modules:")
for name, module in list(base_generator.named_modules())[-5:]:
    print(f'{name}: {module}')
