"""
多序列 MRI 融合模块
支持两种模式:
1. concat: 原始通道拼接模式 (早期融合)
2. transformer: 基于 Transformer 的多序列关系学习模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiSeriesTransformerFusion(nn.Module):
    """
    基于 Transformer 的多序列图像融合模块
    将每个序列视为一个 token，通过自注意力学习序列间关系
    """
    def __init__(
        self,
        num_series: int = 6,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        spatial_size: int = 256
    ):
        """
        Args:
            num_series: 输入序列数量 (默认6: t1, t2, b50, b800, b1500, adc)
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            dropout: Dropout 概率
            spatial_size: 输入图像空间尺寸 (用于位置编码)
        """
        super().__init__()
        self.num_series = num_series
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        
        # 每个序列的独立卷积编码器 (将单通道转换为 embed_dim)
        self.series_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_series)
        ])
        
        # 空间下采样后的尺寸
        self.spatial_tokens_h = spatial_size // 4
        self.spatial_tokens_w = spatial_size // 4
        
        # 序列类型嵌入 (学习每个序列类型的特征)
        self.series_type_embed = nn.Parameter(torch.randn(1, num_series, embed_dim))
        
        # 空间位置编码 (使用 2D 正弦位置编码)
        self.pos_embed = self._create_2d_positional_encoding(
            self.spatial_tokens_h, self.spatial_tokens_w, embed_dim
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影层 (将融合后的特征映射到指定通道数)
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # 最终输出卷积
        self.final_conv = nn.Conv2d(embed_dim // 4, num_series, kernel_size=3, padding=1)
        
    def _create_2d_positional_encoding(self, h: int, w: int, d_model: int):
        """创建 2D 正弦位置编码"""
        pe = torch.zeros(h * w, d_model)
        y_pos = torch.arange(h).repeat_interleave(w).unsqueeze(1).float()
        x_pos = torch.arange(w).repeat(h).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(y_pos * div_term[:pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(x_pos * div_term[:pe[:, 1::2].shape[1]])
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # [1, H*W, D]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
        Returns:
            融合后的特征 [B, num_series, H, W] (保持与输入相同维度以便与原始方案对比)
        """
        B, C, H, W = x.shape
        assert C == self.num_series, f"输入通道数 {C} 与预期 {self.num_series} 不符"
        
        # 对每个序列独立编码
        series_features = []
        for i in range(self.num_series):
            feat = self.series_encoders[i](x[:, i:i+1, :, :])  # [B, embed_dim, H/4, W/4]
            series_features.append(feat)
        
        # 堆叠序列特征: [B, num_series, embed_dim, H/4, W/4]
        series_features = torch.stack(series_features, dim=1)
        
        # 重塑为 Transformer 输入格式
        # [B, num_series, embed_dim, H/4, W/4] -> [B, num_series, H/4*W/4, embed_dim]
        B, S, D, H_s, W_s = series_features.shape
        series_features = series_features.view(B, S, D, -1).permute(0, 1, 3, 2)  # [B, S, H*W, D]
        
        # 添加序列类型嵌入
        series_features = series_features + self.series_type_embed.unsqueeze(2)  # [B, S, H*W, D]
        
        # 添加空间位置编码
        pos_embed = self.pos_embed[:, :H_s*W_s, :].unsqueeze(1)  # [1, 1, H*W, D]
        series_features = series_features + pos_embed
        
        # 合并序列和空间维度作为 token: [B, S*H*W, D]
        tokens = series_features.view(B, -1, D)
        
        # Transformer 编码
        fused_tokens = self.transformer(tokens)  # [B, S*H*W, D]
        
        # 恢复形状: [B, S, H*W, D] -> [B, S, D, H, W]
        fused_features = fused_tokens.view(B, S, H_s * W_s, D).permute(0, 1, 3, 2)
        fused_features = fused_features.view(B, S, D, H_s, W_s)
        
        # 对每个序列的特征进行上采样和投影
        output_features = []
        for i in range(self.num_series):
            feat = self.output_proj(fused_features[:, i, :, :, :])  # [B, embed_dim/4, H, W]
            output_features.append(feat)
        
        # 合并并生成最终输出
        output_features = torch.stack(output_features, dim=1)  # [B, num_series, C', H, W]
        B, S, C_mid, H_out, W_out = output_features.shape
        output_features = output_features.view(B * S, C_mid, H_out, W_out)
        
        # 最终卷积生成加权图
        weights = self.final_conv(output_features)  # [B*S, num_series, H, W]
        weights = weights.view(B, S, S, H_out, W_out)  # [B, S, S, H, W]
        
        # 使用 softmax 生成注意力权重
        weights = F.softmax(weights, dim=2)  # 在序列维度上归一化
        
        # 加权融合: 对每个位置，所有序列的加权平均
        x_expanded = x.unsqueeze(2)  # [B, S, 1, H, W]
        fused_output = (weights * x_expanded).sum(dim=2)  # [B, S, H, W]
        
        # 残差连接
        output = fused_output + x
        
        return output


class MultiSeriesFusionWrapper(nn.Module):
    """
    多序列融合包装器
    支持两种模式切换: concat / transformer
    """
    def __init__(
        self,
        num_series: int = 6,
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
            self.fusion_module = MultiSeriesTransformerFusion(
                num_series=num_series,
                embed_dim=config.get("embed_dim", 256),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 4),
                dropout=config.get("dropout", 0.1),
                spatial_size=config.get("spatial_size", 256)
            )
        elif fusion_mode == "concat":
            # 原始模式: 直接返回输入，不做任何处理
            self.fusion_module = None
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}，请选择 'concat' 或 'transformer'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
        Returns:
            融合后的特征 [B, num_series, H, W]
        """
        if self.fusion_mode == "concat":
            # 原始模式: 直接返回输入
            return x
        else:
            # Transformer 融合模式
            return self.fusion_module(x)
    
    def get_output_channels(self) -> int:
        """获取输出通道数"""
        return self.num_series


class GeneratorWithFusion(nn.Module):
    """
    带有多序列融合模块的生成器
    """
    def __init__(
        self,
        base_generator: nn.Module,
        num_series: int = 6,
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入多序列图像 [B, num_series, H, W]
        Returns:
            生成的 CE-MRI [B, 1, H, W]
        """
        # 多序列融合
        fused_features = self.fusion_module(x)
        
        # 通过生成器生成图像
        output = self.generator(fused_features)
        
        return output


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试 Transformer 融合模块
    print("测试 MultiSeriesTransformerFusion...")
    fusion_module = MultiSeriesTransformerFusion(
        num_series=6,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        spatial_size=256
    ).to(device)
    
    x = torch.randn(2, 6, 256, 256).to(device)
    output = fusion_module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试包装器 - concat 模式
    print("\n测试 MultiSeriesFusionWrapper (concat 模式)...")
    wrapper_concat = MultiSeriesFusionWrapper(num_series=6, fusion_mode="concat")
    output_concat = wrapper_concat(x)
    print(f"输出形状: {output_concat.shape}")
    
    # 测试包装器 - transformer 模式
    print("\n测试 MultiSeriesFusionWrapper (transformer 模式)...")
    wrapper_transformer = MultiSeriesFusionWrapper(
        num_series=6, 
        fusion_mode="transformer",
        transformer_config={"embed_dim": 256, "spatial_size": 256}
    ).to(device)
    output_transformer = wrapper_transformer(x)
    print(f"输出形状: {output_transformer.shape}")
    
    print("\n所有测试通过!")
