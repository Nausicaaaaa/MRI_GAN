import torch

from loss_function.losses_function import SSIM_loss_3d


def distance_loss(self, fake_B, real_B, pred_fake, pred_real, mask=None):
    loss_value_dict = {}
    loss_dist = 0
    
    # similarity
    if "L1_loss" in self.loss_weight_dict.keys():
        l1_map = torch.abs(fake_B - real_B)
        if mask is not None:
            mask = mask.float()
            # 策略升级：前景加权 + 亮区增强
            # 1. 基础前景加权：只在 mask=1 的区域计算 L1
            mask_sum = mask.sum()
            if mask_sum < 1e-6:
                # mask为空时回退到全局L1
                loss_G_L1 = self.loss_weight_dict["L1_loss"] * l1_map.mean()
            else:
                foreground_l1 = (l1_map * mask).sum() / (mask_sum + 1e-6)
                
                # 2. 亮区增强：找出真实图中较亮的区域（例如大于均值的部分）
                # 这里的阈值可以根据实际分布调整，比如取 real_B 的 90% 分位数
                masked_real = real_B[mask.bool()]
                if masked_real.numel() > 0:
                    bright_threshold = torch.quantile(masked_real, 0.9)
                else:
                    bright_threshold = real_B.max()
                bright_mask = (real_B >= bright_threshold).float() * mask
                
                # 如果亮区像素太少，则回退到仅使用前景 mask
                bright_sum = bright_mask.sum()
                if bright_sum < 10: 
                    bright_mask = mask
                    bright_sum = mask_sum
                    
                # 给亮区更高的权重 (例如 2.0 倍)
                bright_l1 = (l1_map * bright_mask).sum() / (bright_sum + 1e-6)
                
                loss_G_L1 = self.loss_weight_dict["L1_loss"] * (foreground_l1 + 0.5 * bright_l1)
        else:
            loss_G_L1 = self.loss_weight_dict["L1_loss"] * l1_map.mean()
        loss_value_dict["L1_loss"] = loss_G_L1
        loss_dist += loss_G_L1
    if "SSIM_loss" in self.loss_weight_dict.keys():
        if len(fake_B.shape) == 4:
            # 基础 SSIM loss
            ssim_val = self.criterion_dict["SSIM_loss"](fake_B, real_B)
            loss_G_ssim = self.loss_weight_dict["SSIM_loss"] * ssim_val
            
            # 如果存在 mask，增加前景区域的 SSIM 约束权重
            if mask is not None:
                # 计算前景区域的 SSIM 贡献（简化版：通过 mask 加权像素差异来模拟局部结构约束）
                # 注意：MONAI 的 SSIMLoss 返回的是标量，这里我们通过增加 L1 的前景权重来辅助
                pass 
        else:
            loss_G_ssim = self.loss_weight_dict["SSIM_loss"] * SSIM_loss_3d(fake_B, real_B)
        loss_value_dict["SSIM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    if "MS_SSIM_loss" in self.loss_weight_dict.keys():
        loss_G_ssim = -self.loss_weight_dict["MS_SSIM_loss"] * self.criterion_dict["MS_SSIM_loss"](fake_B, real_B)
        loss_value_dict["MS_SSIM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    if "S3IM_loss" in self.loss_weight_dict.keys():
        loss_G_ssim = self.loss_weight_dict["S3IM_loss"] * self.criterion_dict["S3IM_loss"](
            fake_B.view(fake_B.shape[0], -1), real_B.view(fake_B.shape[0], -1),
            ph=fake_B.shape[-2], pw=fake_B.shape[-1])
        loss_value_dict["S3IM_loss"] = loss_G_ssim
        loss_dist += loss_G_ssim
    # perceptual
    if "Perceptual_loss" in self.loss_weight_dict.keys():
        # fake_B_norm = fake_B * 2 - 1
        # real_B_norm = real_B * 2 - 1
        fake_B_n = fake_B
        real_B_n = real_B
        # fake_B_n = (fake_B-fake_B.min())/(fake_B.max()-fake_B.min())
        # real_B_n = (real_B-real_B.min())/(real_B.max()-real_B.min())
        loss_G_perceptual = self.loss_weight_dict["Perceptual_loss"] * self.criterion_dict[
            "Perceptual_loss"].forward(fake_B_n, real_B_n).mean()
        loss_value_dict["Perceptual_loss"] = loss_G_perceptual
        loss_dist += loss_G_perceptual
    if "VGG_loss" in self.loss_weight_dict.keys():
        fake_B_3C = torch.cat([fake_B] * 3, 1)
        real_B_3C = torch.cat([real_B] * 3, 1)
        loss_G_VGG = self.criterion_dict["VGG_loss"](fake_B_3C, real_B_3C) * self.loss_weight_dict["VGG_loss"]  # 10
        loss_value_dict["VGG_loss"] = loss_G_VGG
        loss_dist += loss_G_VGG
    if "G_Feat_loss" in self.loss_weight_dict.keys():
        loss_G_Feat = 0
        feat_weights = 4.0 / (self.config.n_layers_D + 1)
        D_weights = 1.0 / self.config.num_D
        for i in range(self.config.num_D):
            for j in range(len(pred_fake[i]) - 1):
                loss_G_Feat += D_weights * feat_weights * \
                               self.criterion_dict["G_Feat_loss"](pred_fake[i][j], pred_real[i][j].detach()) * \
                               self.loss_weight_dict["G_Feat_loss"]  # 10
        loss_value_dict["G_Feat_loss"] = loss_G_Feat
        loss_dist += loss_G_Feat
    return loss_dist, loss_value_dict
