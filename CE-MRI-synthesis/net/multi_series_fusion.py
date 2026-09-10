"""
多序列 MRI 融合模块
支持两种模式:
1. concat: 原始通道拼接模式 (早期融合)
2. transformer: 基于 Transformer 的像素级模态注意力融合模式

核心改进（2024版）:
- 像素级加权：每个像素位置都有独立的序列权重
- output[b, s, h, w] = input[b, s, h, w] * weight[b, s, h, w]
- 保留原始分辨率，适合肿瘤识别等精细结构任务
- 参数量仅 ~2,353 个，计算开销极低
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from contextlib import nullcontext


class PixelWiseModalAttention(nn.Module):
    """
    像素级模态注意力模块（推荐版本）
    
    核心设计：
    - 为每个像素位置生成独立的序列注意力权重 [B, num_series, H, W]
    - 通过轻量级卷积+Transformer实现，不修改原始图像值
    - output[b, s, h, w] = input[b, s, h, w] * weight[b, s, h, w]
    - 空间池化降低计算量，双线性上采样恢复原始分辨率
    
    优势：
    - 空间自适应：不同区域自动选择重要序列（如肿瘤区域关注DWI）
    - 保留完整分辨率：无下采样损失细节
    - 参数量极小：~2,353 参数
    - 可解释性强：权重大小直接反映序列重要性
    
    数据流：
    输入 [B, S, H, W]
      → 空间池化 [B, S, pool, pool]
      → 输入投影 [B, S, D, pool, pool]
      → Transformer 学习模态关系 (序列长度=S=5)
      → 输出投影 [B, S, 1, pool, pool] -> 权重
      → 上采样 [B, S, 1, H, W] -> 权重
      → 逐像素加权: output = input * weight
    """
    def __init__(
        self,
        num_series: int = 5,
        embed_dim: int = 24,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.1,
        spatial_pool_size: int = 64
    ):
        """
        Args:
            num_series: 输入序列数量 (默认5: pret1, t1, t2, dwi, adc)
            embed_dim: 嵌入维度（用于学习模态关系的特征维度）
            num_heads: 注意力头数
            num_layers: Transformer 层数
            dropout: Dropout 概率
            spatial_pool_size: 池化尺寸（平衡计算量和细节保留）
        """
        super().__init__()
        self.num_series = num_series
        self.embed_dim = embed_dim
        self.spatial_pool_size = spatial_pool_size
        
        # 空间池化：降低计算量
        self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
        
        # 输入投影：[B*S, 1, pool, pool] -> [B*S, D, pool, pool]
        self.input_proj = nn.Conv2d(1, embed_dim, kernel_size=1)
        
        # 模态类型嵌入（可学习参数，区分不同模态）
        self.modal_type_embed = nn.Parameter(torch.randn(1, num_series, embed_dim))
        
        # 轻量级 Transformer Encoder（序列长度=num_series，非常小）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影：[B*S, D, pool, pool] -> [B*S, 1, pool, pool] (注意力权重)
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()  # 输出 [0, 1] 范围的权重
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
            return_attention: 是否返回注意力权重热图
        Returns:
            注意力加权后的特征 [B, num_series, H, W]
            如果 return_attention=True，额外返回 attn_weights [B, num_series, H, W]
            每个像素位置的输出 = 输入 * 权重（逐像素自适应加权）
        """
        B, C, H, W = x.shape
        assert C == self.num_series, f"输入通道数 {C} 与预期 {self.num_series} 不符"
        
        pool_size = self.spatial_pool_size
        
        # 1. 空间池化: [B, S, H, W] -> [B, S, pool, pool]
        # 将每个序列视为独立的batch处理
        x_pooled = self.spatial_pool(x.view(B * C, 1, H, W))
        x_pooled = x_pooled.view(B, C, pool_size, pool_size)
        
        # 2. 输入投影: [B, S, pool, pool] -> [B, S, D, pool, pool]
        x_proj = self.input_proj(x_pooled.view(B * C, 1, pool_size, pool_size))
        x_proj = x_proj.view(B, C, self.embed_dim, pool_size, pool_size)
        
        # 3. 添加模态类型嵌入
        x_proj = x_proj + self.modal_type_embed.view(1, C, self.embed_dim, 1, 1)
        
        # 4. 重排为 [B, pool, pool, S, D] -> [B*pool*pool, S, D]
        # 将空间位置作为batch维度，Transformer仅处理模态关系
        x_proj = x_proj.permute(0, 3, 4, 1, 2).contiguous()
        x_flat = x_proj.reshape(B * pool_size * pool_size, C, self.embed_dim)
        
        # 5. Transformer 编码（序列长度仅为 S=5，计算量极小）
        # 分块处理，避免 spatial_pool_size 过大时 scaled_dot_product_attention 的 CUDA batch 维度溢出
        _chunk = 4096
        if x_flat.shape[0] > _chunk:
            # 转为普通 Tensor，避免 MONAI MetaTensor 切片时元数据冲突
            x_plain = x_flat.as_tensor() if hasattr(x_flat, 'as_tensor') else x_flat
            x_transformed = torch.cat(
                [self.transformer(x_plain[i:i + _chunk]) for i in range(0, x_plain.shape[0], _chunk)],
                dim=0
            )
        else:
            x_transformed = self.transformer(x_flat)
        
        # 6. 恢复空间维度: [B*pool*pool, S, D] -> [B, S, D, pool, pool]
        x_out = x_transformed.reshape(B, pool_size, pool_size, C, self.embed_dim)
        x_out = x_out.permute(0, 3, 4, 1, 2).contiguous()
        x_out = x_out.reshape(B * C, self.embed_dim, pool_size, pool_size)
        
        # 7. 输出投影: [B*S, D, pool, pool] -> [B*S, 1, pool, pool]
        attn_weights = self.output_proj(x_out)
        attn_weights = attn_weights.view(B, C, pool_size, pool_size)
        
        # 8. 上采样回原始分辨率: [B, S, pool, pool] -> [B, S, H, W]
        # 使用双线性插值，确保平滑的权重图
        attn_weights = F.interpolate(
            attn_weights, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=True
        )
        
        # 9. 像素级加权：output = input * weight
        # 关键：不修改原始图像值，只做自适应加权
        output = x * attn_weights
        
        if return_attention:
            return output, attn_weights
        return output


class MultiSeriesFusionWrapper(nn.Module):
    """
    多序列融合包装器
    支持两种模式切换: concat / transformer
    """
    def __init__(
        self,
        num_series: int = 5,
        fusion_mode: str = "concat",
        transformer_config: dict = None
    ):
        """
        Args:
            num_series: 输入序列数量
            fusion_mode: 融合模式 ("concat" 或 "transformer")
            transformer_config: Transformer 配置参数
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        self.num_series = num_series
        
        if fusion_mode == "transformer":
            config = transformer_config or {}
            # 使用像素级模态注意力模块
            self.fusion_module = PixelWiseModalAttention(
                num_series=num_series,
                embed_dim=config.get("embed_dim", 24),
                num_heads=config.get("num_heads", 2),
                num_layers=config.get("num_layers", 1),
                dropout=config.get("dropout", 0.1),
                spatial_pool_size=config.get("spatial_pool_size", 64)
            )
        elif fusion_mode == "concat":
            # 原始模式: 直接返回输入，不做任何处理
            self.fusion_module = None
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}，请选择 'concat' 或 'transformer'")
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
            return_attention: 是否返回注意力权重热图
        Returns:
            融合后的特征 [B, num_series, H, W]
            如果 return_attention=True，额外返回 attn_weights [B, num_series, H, W]
        """
        if self.fusion_mode == "concat":
            # 原始模式: 直接返回输入
            if return_attention:
                return x, None
            return x
        else:
            # Transformer 融合模式：像素级加权
            return self.fusion_module(x, return_attention=return_attention)
    
    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.num_series


class GeneratorWithFusion(nn.Module):
    """
    带有多序列融合模块的生成器
    
    工作流程:
    1. 多序列输入 [B, num_series, H, W]
    2. 融合模块处理（可选）：
       - concat模式: 直接传递，不做处理
       - transformer模式: 像素级加权 output = input * weight
    3. 生成器生成最终图像
    """
    def __init__(
        self,
        base_generator: nn.Module,
        num_series: int = 5,
        fusion_mode: str = "concat",
        transformer_config: dict = None
    ):
        """
        Args:
            base_generator: 基础生成器 (如 UNet)
            num_series: 输入序列数量
            fusion_mode: 融合模式
            transformer_config: Transformer 配置
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        
        # 多序列融合模块
        self.fusion_module = MultiSeriesFusionWrapper(
            num_series=num_series,
            fusion_mode=fusion_mode,
            transformer_config=transformer_config
        )
        
        # 基础生成器
        self.generator = base_generator
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
            return_attention: 是否返回注意力权重热图
        Returns:
            生成的 CE-MRI [B, 1, H, W]
            如果 return_attention=True，额外返回 attn_weights [B, num_series, H, W]
        """
        # 多序列融合
        if return_attention:
            fused_features, attn_weights = self.fusion_module(x, return_attention=True)
        else:
            fused_features = self.fusion_module(x)
            attn_weights = None
        
        # 通过生成器生成图像
        output = self.generator(fused_features)
        
        if return_attention:
            return output, attn_weights
        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试 PixelWiseModalAttention 模块
    print("测试 PixelWiseModalAttention...")
    fusion_module = PixelWiseModalAttention(
        num_series=5,
        embed_dim=24,
        num_heads=2,
        num_layers=1,
        spatial_pool_size=64
    ).to(device)
    
    # 使用较小尺寸测试，避免显存问题
    test_size = 64
    x = torch.randn(1, 5, test_size, test_size).to(device)
    
    with torch.no_grad():
        output = fusion_module(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in fusion_module.parameters()):,}")
    
    # 验证空间分辨率保持不变
    assert output.shape == x.shape, f"形状不匹配: {output.shape} vs {x.shape}"
    print("✓ 空间分辨率保持不变")
    
    # 验证像素级自适应特性
    print("\n验证像素级自适应特性...")
    # 不同位置的权重应该不同
    w1 = output[0, :, 10, 10] / (x[0, :, 10, 10] + 1e-8)
    w2 = output[0, :, 30, 30] / (x[0, :, 30, 30] + 1e-8)
    
    print(f"位置(10,10)的序列权重: {w1.cpu().numpy()}")
    print(f"位置(30,30)的序列权重: {w2.cpu().numpy()}")
    
    if not torch.allclose(w1, w2, atol=1e-3):
        print("✓ 成功：不同位置有不同的权重（像素级自适应）")
    else:
        print("✗ 警告：权重相同，可能未正确学习空间自适应")
    
    # 测试包装器 - concat 模式
    print("\n测试 MultiSeriesFusionWrapper (concat 模式)...")
    wrapper_concat = MultiSeriesFusionWrapper(num_series=5, fusion_mode="concat")
    output_concat = wrapper_concat(x)
    print(f"输出形状: {output_concat.shape}")
    assert output_concat.shape == x.shape
    
    # 测试包装器 - transformer 模式
    print("\n测试 MultiSeriesFusionWrapper (transformer 模式)...")
    wrapper_transformer = MultiSeriesFusionWrapper(
        num_series=5, 
        fusion_mode="transformer",
        transformer_config={"embed_dim": 24, "spatial_pool_size": 64}
    ).to(device)
    
    with torch.no_grad():
        output_transformer = wrapper_transformer(x)
    
    print(f"输出形状: {output_transformer.shape}")
    print(f"参数量: {sum(p.numel() for p in wrapper_transformer.parameters()):,}")
    assert output_transformer.shape == x.shape
    
    print("\n✓ 所有测试通过!")
