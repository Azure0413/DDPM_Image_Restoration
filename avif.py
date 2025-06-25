import torch
import torchvision
import pillow_avif
from torch import nn
from torch.utils.data import DataLoader, random_split
from PIL import Image
import io 
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
import numpy as np
from pytorch_msssim import ssim
import lpips

# 設置GPU裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 資料集準備
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),  # 調整所有圖像為128×128
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

import os
from torch.utils.data import Dataset
from PIL import Image

class ImageFolderFlat(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 獲取所有圖像文件
        self.image_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 返回0作為虛擬標籤，因為我們不需要真實類別
        return image, 0

# 載入資料集的代碼修改為
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),  # 調整所有圖像為64×64
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolderFlat(
    root="./ILSVRC2012_img_val",  # 請替換為您實際的路徑
    transform=transform
)

# 保持原有的train/valid/test比例
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_size, valid_size, test_size]
)

# 設置資料載入器 - 注意：您可能需要減少batch_size以適應更大的圖像
batch_size = 8  # 減小批次大小以應對更大的圖像
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 定義更精確的AVIF壓縮函數
def avif_compress(x, quality):
    """執行AVIF壓縮並返回解碼結果"""
    # 從[-1,1]轉換為[0,255] uint8
    x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu()
    
    compressed_images = []
    for img in x:
        # 轉換為PIL圖像
        pil_img = torchvision.transforms.ToPILImage()(img)
        
        # 壓縮為AVIF
        buffer = io.BytesIO()
        quality = max(1, min(100, int(quality)))
        
        # AVIF特定參數 - 根據品質動態調整
        speed = 6 if quality > 50 else 4 if quality > 20 else 2
        
        try:
            # 使用AVIF格式保存，配置高級參數
            pil_img.save(buffer, format="AVIF", 
                        quality=quality,
                        speed=speed,  # AVIF編碼速度
                        range='full',  # 色彩範圍
                        subsampling='4:4:4' if quality > 50 else '4:2:0')  # 子採樣
            buffer.seek(0)
            
            # 解碼AVIF
            compressed_img = Image.open(buffer)
            compressed_tensor = torchvision.transforms.ToTensor()(compressed_img)
            compressed_images.append(compressed_tensor)
        except Exception as e:
            # 如果AVIF不支援，回退到高品質JPEG
            print(f"AVIF encoding failed, falling back to JPEG: {e}")
            buffer = io.BytesIO()
            subsampling = "4:4:4" if quality > 30 else "4:2:0"
            pil_img.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed_tensor = torchvision.transforms.ToTensor()(compressed_img)
            compressed_images.append(compressed_tensor)
    
    # 轉換回[-1,1]範圍並返回到設備
    return torch.stack(compressed_images).to(device).sub(0.5).mul(2.0)

# 定義AVIF感知的頻率領域損失
def avif_frequency_aware_loss(pred, target):
    """結合AVIF特性的頻率感知損失函數"""
    # 空間域MSE
    spatial_loss = F.mse_loss(pred, target)
    
    # 轉換到[0,1]範圍進行計算
    pred_01 = pred * 0.5 + 0.5
    target_01 = target * 0.5 + 0.5
    
    # AVIF更好的細節保持 - 增加邊緣保持損失
    def gradient_loss(x, y):
        grad_x_x = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        grad_x_y = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y_x = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        grad_y_y = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        
        return F.mse_loss(grad_x_x, grad_y_x) + F.mse_loss(grad_x_y, grad_y_y)
    
    edge_loss = gradient_loss(pred_01, target_01)
    
    # 頻率域損失 - AVIF使用更先進的變換，不只是DCT
    freq_loss = 0
    for c in range(3):
        # 計算2D FFT (AVIF使用更複雜的變換)
        pred_fft = torch.fft.fft2(pred_01[:, c])
        target_fft = torch.fft.fft2(target_01[:, c])
        
        # 頻率域的MSE（幅度）
        freq_mse = F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        # 相位損失
        phase_loss = F.mse_loss(torch.angle(pred_fft), torch.angle(target_fft))
        
        freq_loss += freq_mse + 0.3 * phase_loss
    
    # SSIM感知損失
    ssim_loss = 1.0 - ssim(pred_01, target_01, data_range=1.0, size_average=True)
    
    # AVIF特定權重調整 - 更注重細節和色彩保持
    return spatial_loss + 0.3 * freq_loss + 0.4 * ssim_loss + 0.2 * edge_loss

# 時間嵌入模組
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.proj(emb)

# AVIF自適應變換層 - 替代固定的DCT
class AVIFAdaptiveTransform(nn.Module):
    """實現可學習的變換操作，模擬AVIF的先進編碼"""
    def __init__(self, channels, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.channels = channels
        
        # 可學習的變換核心
        self.transform_weights = nn.Parameter(torch.randn(channels, block_size, block_size))
        self.inverse_weights = nn.Parameter(torch.randn(channels, block_size, block_size))
        
        # 自適應量化
        self.quantization = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 填充至block_size的整數倍
        h_pad = (self.block_size - h % self.block_size) % self.block_size
        w_pad = (self.block_size - w % self.block_size) % self.block_size
        
        x_padded = F.pad(x, (0, w_pad, 0, h_pad))
        h_padded, w_padded = x_padded.shape[2], x_padded.shape[3]
        
        # 分割成塊
        patches = x_padded.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        patches = patches.contiguous().view(b, c, -1, self.block_size, self.block_size)
        
        # 應用可學習變換
        transformed_patches = []
        for ch in range(c):
            ch_patches = patches[:, ch]  # [b, num_blocks, block_size, block_size]
            transform_matrix = self.transform_weights[ch]
            
            # 執行變換: T * X * T^T
            transformed = torch.matmul(torch.matmul(transform_matrix, ch_patches), transform_matrix.transpose(0, 1))
            transformed_patches.append(transformed)
        
        transformed_patches = torch.stack(transformed_patches, dim=1)
        
        # 重構回空間域
        num_blocks_h = h_padded // self.block_size
        num_blocks_w = w_padded // self.block_size
        
        transformed_spatial = transformed_patches.view(b, c, num_blocks_h, num_blocks_w, 
                                                     self.block_size, self.block_size)
        transformed_spatial = transformed_spatial.permute(0, 1, 2, 4, 3, 5).contiguous()
        transformed_spatial = transformed_spatial.view(b, c, h_padded, w_padded)
        
        # 移除填充
        if h_pad > 0 or w_pad > 0:
            transformed_spatial = transformed_spatial[:, :, :h, :w]
        
        # 應用自適應量化
        quantized = self.quantization(transformed_spatial)
        
        return transformed_spatial * quantized

# AVIF頻率感知塊
class AVIFFreqAwareBlock(nn.Module):
    """專門為AVIF特性設計的頻率感知模塊"""
    def __init__(self, channels, block_size=8):
        super().__init__()
        self.block_size = block_size
        self.adaptive_transform = AVIFAdaptiveTransform(channels, block_size)
        
        # 多尺度頻率注意力 - AVIF能更好處理不同尺度
        self.multi_scale_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            ) for scale in [1, 2, 4, 8]
        ])
        
        # 色彩一致性模組 - AVIF支援寬色域
        self.color_consistency = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
        # 邊緣保持模組
        self.edge_preserve = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 輸出層
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x, compression_level=None):
        # 自適應變換處理
        x_transformed = self.adaptive_transform(x)
        
        # 多尺度注意力
        attn_sum = 0
        for attn_module in self.multi_scale_attn:
            attn = attn_module(x)
            if attn.shape != x.shape:
                attn = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
            attn_sum += attn
        
        attn_avg = attn_sum / len(self.multi_scale_attn)
        
        # 色彩一致性
        color_attn = self.color_consistency(x)
        
        # 邊緣保持
        edge_attn = self.edge_preserve(x)
        
        # 基於壓縮級別調整注意力
        if compression_level is not None:
            if isinstance(compression_level, torch.Tensor) and compression_level.dim() > 0:
                compression_level = compression_level.view(-1, 1, 1, 1)
            # AVIF在低品質時仍能保持較好的色彩和邊緣
            color_boost = torch.clamp(0.5 + 0.5 * (1.0 - compression_level), 0.3, 1.5)
            edge_boost = torch.clamp(0.7 + 0.3 * (1.0 - compression_level), 0.5, 1.3)
            
            color_attn = color_attn * color_boost
            edge_attn = edge_attn * edge_boost
        
        # 組合所有特徵
        enhanced = x_transformed * attn_avg * color_attn * edge_attn
        
        # 殘差連接
        return self.conv_out(x + enhanced)

# 改進的殘差注意力塊，整合AVIF頻率感知
class AVIFResAttnBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim, dropout=0.1):
        super().__init__()
        # 確保組數適合通道數
        num_groups = min(8, in_c)
        while in_c % num_groups != 0 and num_groups > 1:
            num_groups -= 1
            
        self.norm1 = nn.GroupNorm(num_groups, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_c)
        
        # 調整out_c的組數
        num_groups_out = min(8, out_c)
        while out_c % num_groups_out != 0 and num_groups_out > 1:
            num_groups_out -= 1
            
        self.norm2 = nn.GroupNorm(num_groups_out, out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        
        # 自注意力機制 - 增加頭數以處理AVIF的複雜性
        self.attn = nn.MultiheadAttention(out_c, 8, batch_first=True)
        
        # AVIF頻率感知處理
        self.freq_guide = AVIFFreqAwareBlock(out_c)
        
        # 殘差連接
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        
    def forward(self, x, t_emb, compression_level=None):
        h = self.norm1(x)
        h = self.conv1(h)
        
        # 加入時間嵌入
        t = self.time_proj(t_emb)[..., None, None]
        h = h + t
        
        h = self.norm2(h)
        h = F.gelu(h)  # 使用GELU激活函數
        h = self.dropout(h)
        h = self.conv2(h)
        
        # 應用自注意力
        b, c, height, width = h.shape
        h_flat = h.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        h_attn, _ = self.attn(h_flat, h_flat, h_flat)
        h_attn = h_attn.permute(0, 2, 1).view(b, c, height, width)
        h = h + h_attn
        
        # 應用AVIF頻率感知處理
        h = self.freq_guide(h, compression_level)
        
        # 殘差連接
        return self.shortcut(x) + h

# 完整的UNet架構，專為AVIF偽影去除設計
class AVIFDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_embed = TimeEmbedding(time_dim)
        
        # 下採樣路徑
        self.down1 = AVIFResAttnBlock(3, 64, time_dim)
        self.down2 = AVIFResAttnBlock(64, 128, time_dim)
        self.down3 = AVIFResAttnBlock(128, 256, time_dim)
        self.down4 = AVIFResAttnBlock(256, 512, time_dim)
        self.down5 = AVIFResAttnBlock(512, 512, time_dim)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶頸層
        self.bottleneck = nn.Sequential(
            AVIFResAttnBlock(512, 1024, time_dim),
            AVIFResAttnBlock(1024, 1024, time_dim),
            AVIFResAttnBlock(1024, 512, time_dim)
        )
        
        # 上採樣路徑
        self.up1 = AVIFResAttnBlock(1024, 512, time_dim)
        self.up2 = AVIFResAttnBlock(1024, 256, time_dim)
        self.up3 = AVIFResAttnBlock(512, 128, time_dim)
        self.up4 = AVIFResAttnBlock(256, 64, time_dim)
        self.up5 = AVIFResAttnBlock(128, 64, time_dim)
        
        # AVIF感知層
        self.avif_layer = AVIFAdaptiveTransform(64, block_size=8)
        
        # 輸出層
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x, t, compression_level=None):
        t_emb = self.time_embed(t)
        
        # 若未提供壓縮級別，使用t值
        if compression_level is None:
            compression_level = t.clone().detach()
        
        # 下採樣路徑
        d1 = self.down1(x, t_emb, compression_level)
        d2 = self.down2(self.pool(d1), t_emb, compression_level)
        d3 = self.down3(self.pool(d2), t_emb, compression_level)
        d4 = self.down4(self.pool(d3), t_emb, compression_level)
        d5 = self.down5(self.pool(d4), t_emb, compression_level)
        
        # 瓶頸層
        bottleneck = self.bottleneck[0](self.pool(d5), t_emb, compression_level)
        bottleneck = self.bottleneck[1](bottleneck, t_emb, compression_level)
        bottleneck = self.bottleneck[2](bottleneck, t_emb, compression_level)
        
        # 上採樣路徑，添加跳躍連接
        u1 = self.up1(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False), d5], dim=1), t_emb, compression_level)
        u2 = self.up2(torch.cat([F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False), d4], dim=1), t_emb, compression_level)
        u3 = self.up3(torch.cat([F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False), d3], dim=1), t_emb, compression_level)
        u4 = self.up4(torch.cat([F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False), d2], dim=1), t_emb, compression_level)
        u5 = self.up5(torch.cat([F.interpolate(u4, scale_factor=2, mode='bilinear', align_corners=False), d1], dim=1), t_emb, compression_level)
        
        # 應用AVIF感知層增強特徵
        avif_feature = self.avif_layer(u5)
        combined = u5 + 0.15 * avif_feature  # 增強AVIF特徵融合
        
        return self.out_conv(combined)

# 相位一致性函數 - 保持圖像結構特徵，針對AVIF優化
def phase_consistency(x, ref, alpha=0.8):
    """使用傅里葉變換的相位一致性，針對AVIF的特性優化"""
    # FFT變換
    x_fft = torch.fft.fft2(x)
    ref_fft = torch.fft.fft2(ref)
    
    # 獲取幅度和相位
    x_mag = torch.abs(x_fft)
    ref_phase = torch.angle(ref_fft)
    
    # 融合新的複數值，使用x的幅度和參考的相位
    real = x_mag * torch.cos(ref_phase)
    imag = x_mag * torch.sin(ref_phase)
    adjusted_fft = torch.complex(real, imag)
    
    # 逆變換
    adjusted_img = torch.fft.ifft2(adjusted_fft).real
    
    # 混合原始圖像和相位調整圖像
    return alpha * x + (1 - alpha) * adjusted_img

# DDRM-AVIF採樣器 - 核心採樣邏輯
class DDRMAVIFSampler:
    def __init__(self, model):
        self.model = model
        
    def sample(self, x_t, quality, steps=100, eta=0.85, eta_b=1.0):
        """DDRM-AVIF採樣方法，專為AVIF偽影去除設計"""
        self.model.eval()
        
        # 保存原始壓縮圖像作為測量值y
        y = x_t.clone()
        
        with torch.no_grad():
            # 反向擴散過程
            for i in range(steps-1, -1, -1):
                # 計算標準化時間步
                t = torch.full((x_t.size(0),), i, device=device).float() / steps
                
                # 下一個時間步（用於噪聲縮放）
                t_next = torch.full((x_t.size(0),), max(0, i-1), device=device).float() / steps
                
                # 壓縮級別與時間步關聯
                compression_level = t.clone()
                
                # 模型預測
                x_theta = self.model(x_t, t, compression_level)
                
                # DDRM-AVIF更新規則
                # 首先，對預測結果進行AVIF壓縮
                avif_x_theta = avif_compress(x_theta, quality)
                
                # 根據DDRM-AVIF公式計算校正項
                x_prime = x_theta - avif_x_theta + y
                
                if i > 0:
                    # 計算噪聲 - AVIF通常品質更好，使用較小噪聲
                    noise_scale = t.float() * 0.15  # 降低噪聲尺度
                    random_noise = torch.randn_like(x_t) * noise_scale.view(-1, 1, 1, 1)
                    
                    # 混合校正項、預測和噪聲
                    x_t = eta_b * x_prime + (1 - eta_b) * x_theta + eta * random_noise
                    
                    # 低質量AVIF的額外穩定處理
                    if quality < 30 and i % 3 == 0:  # AVIF在低品質時表現更好
                        # 應用相位一致性以保留邊緣和色彩
                        x_t = phase_consistency(x_t, y, alpha=0.8)
                else:
                    # 最後一步 - 只使用校正後的預測
                    x_t = x_prime
        
        return x_t

# 更新訓練函數
def train_epoch_ddrm_avif(model, loader, epoch, optimizer, scheduler):
    model.train()
    total_loss = 0
    freq_loss_total = 0
    ssim_loss_total = 0
    
    for x0, _ in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        x0 = x0.to(device)
        b = x0.size(0)
        
        # AVIF質量選擇策略 - 自適應增加高質量比例
        epoch_progress = min(1.0, epoch / 100)  # 標準化到[0,1]
        if random.random() < 0.3 + 0.4 * epoch_progress:
            # 高質量 - AVIF高品質範圍更廣
            quality_range = (75, 100)
        elif random.random() < 0.5:
            # 中等質量
            quality_range = (45, 75)
        else:
            # 低質量 - AVIF在低品質時仍有不錯表現
            quality_range = (10, 45)
            
        # 隨機時間步選擇
        t = torch.randint(1, steps, (b,), device=device).long()
        
        # 基於時間步計算每個樣本的質量
        min_q, max_q = quality_range
        quality = torch.clamp(min_q + (max_q - min_q) * (1 - t.float() / steps), 1, 100).cpu().numpy()
        
        # 應用AVIF壓縮獲取帶噪聲圖像
        xt = torch.stack([avif_compress(x0[i:i+1], int(q)) for i, q in enumerate(quality)])
        if xt.dim() > 4:  # 處理批次維度被擴展的情況
            xt = xt.squeeze(1)
        
        # 計算目標（噪聲/殘差）
        target = x0 - xt
        
        # 獲取模型預測
        compression_level = t.float() / steps
        pred = model(xt, t.float()/steps, compression_level)
        
        # 計算AVIF感知頻率損失
        loss = avif_frequency_aware_loss(xt + pred, x0)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 跟踪損失
        total_loss += loss.item()
        
    # 更新學習率
    scheduler.step()
    
    # 報告指標
    avg_loss = total_loss / len(loader)
    
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.5f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    return avg_loss

# 驗證函數
def validate_ddrm_avif(model, loader, epoch):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        for x0, _ in tqdm(loader, desc=f"Validating Epoch {epoch+1}"):
            x0 = x0.to(device)
            b = x0.size(0)
            
            # 選擇多種質量進行驗證
            qualities = [20, 50, 80]  # AVIF品質範圍調整
            
            for quality in qualities:
                # 創建壓縮圖像
                y = avif_compress(x0, quality)
                
                # 設置初始時間步與質量相關
                init_t = int((100 - quality) / 100 * steps)
                init_t = max(15, min(init_t, 75))  # AVIF調整範圍
                
                # 使用採樣器恢復
                sampler = DDRMAVIFSampler(model)
                restored = sampler.sample(y, quality, steps=init_t)
                
                # 計算指標
                x0_01 = (x0 * 0.5 + 0.5).clamp(0, 1)
                y_01 = (y * 0.5 + 0.5).clamp(0, 1)
                restored_01 = (restored * 0.5 + 0.5).clamp(0, 1)
                
                # PSNR
                mse = F.mse_loss(restored_01, x0_01).item()
                psnr = -10 * math.log10(mse)
                
                # SSIM
                ssim_val = ssim(restored_01, x0_01, data_range=1.0).item()
                
                # LPIPS
                lpips_val = lpips_model(restored_01 * 2 - 1, x0_01 * 2 - 1).mean().item()
                
                total_psnr += psnr
                total_ssim += ssim_val
                total_lpips += lpips_val
    
    # 計算平均值
    num_evals = len(loader) * len(qualities)
    avg_psnr = total_psnr / num_evals
    avg_ssim = total_ssim / num_evals
    avg_lpips = total_lpips / num_evals
    
    print(f"Validation - PSNR: {avg_psnr:.2f}dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    
    # 可視化一些結果
    if epoch % 5 == 0:
        visualize_avif_restoration(model, epoch)
    
    return avg_psnr, avg_ssim, avg_lpips

# 可視化結果函數
def visualize_avif_restoration(model, epoch):
    model.eval()
    sampler = DDRMAVIFSampler(model)
    
    with torch.no_grad():
        x0, _ = next(iter(test_dataloader))
        x0 = x0.to(device)
        
        # 測試不同的質量級別
        qualities = [10, 20, 50, 80]  # AVIF品質範圍
        plt.figure(figsize=(len(qualities)*3+3, 5))
        
        # 顯示原始圖像
        plt.subplot(2, len(qualities)+1, 1)
        plt.imshow(x0[0].cpu().permute(1,2,0)*0.5+0.5)
        plt.title("Original")
        plt.axis('off')
        
        # 對每個質量級別顯示AVIF和還原結果
        for i, q in enumerate(qualities):
            # AVIF壓縮
            y = avif_compress(x0, q)
            
            # 設定初始時間步長對應質量
            init_t = int((100 - q) / 100 * steps)
            init_t = max(15, min(init_t, 75))  # AVIF調整範圍
            
            # 使用採樣器進行還原
            restored = sampler.sample(y, q, steps=init_t)
            
            # 計算PSNR
            x0_01 = (x0 * 0.5 + 0.5).clamp(0, 1)
            y_01 = (y * 0.5 + 0.5).clamp(0, 1)
            restored_01 = (restored * 0.5 + 0.5).clamp(0, 1)
            
            y_psnr = -10 * math.log10(F.mse_loss(y_01, x0_01).item())
            restored_psnr = -10 * math.log10(F.mse_loss(restored_01, x0_01).item())
            
            # 顯示AVIF壓縮結果
            plt.subplot(2, len(qualities)+1, i+2)
            plt.imshow(y[0].cpu().permute(1,2,0)*0.5+0.5)
            plt.title(f"AVIF Q{q}\nPSNR: {y_psnr:.2f}dB")
            plt.axis('off')
            
            # 顯示還原結果
            plt.subplot(2, len(qualities)+1, len(qualities)+i+2)
            plt.imshow(restored[0].cpu().permute(1,2,0)*0.5+0.5)
            plt.title(f"Restored\nPSNR: {restored_psnr:.2f}dB")
            plt.axis('off')
        
        plt.tight_layout()
        os.makedirs("./avif_viz", exist_ok=True)
        plt.savefig(f'./avif_viz/avif_restoration_epoch_{epoch}.png')
        plt.close()

# 完整測試函數
def test_avif_restoration(model, quality_levels=[10, 20, 50, 80]):
    # 初始化採樣器
    sampler = DDRMAVIFSampler(model)
    model.eval()
    
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    with torch.no_grad():
        # 對每個質量級別測試
        results = {q: {'psnr': [], 'ssim': [], 'lpips': []} for q in quality_levels}
        
        for idx in tqdm(range(100), desc="Testing"):
            # 選擇測試圖像
            x0, _ = next(iter(test_dataloader))
            x0 = x0.to(device)
            
            for q in quality_levels:
                # AVIF壓縮
                y = avif_compress(x0, q)
                
                # 設定初始時間步長對應質量
                init_t = int((100 - q) / 100 * steps)
                init_t = max(15, min(init_t, 75))
                
                # 使用採樣器進行還原
                restored = sampler.sample(y, q, steps=init_t)
                
                # 計算指標
                x0_01 = (x0 * 0.5 + 0.5).clamp(0, 1)
                y_01 = (y * 0.5 + 0.5).clamp(0, 1)
                restored_01 = (restored * 0.5 + 0.5).clamp(0, 1)
                
                # PSNR
                y_psnr = -10 * math.log10(F.mse_loss(y_01, x0_01).item())
                restored_psnr = -10 * math.log10(F.mse_loss(restored_01, x0_01).item())
                
                # SSIM
                y_ssim = ssim(y_01, x0_01, data_range=1.0).item()
                restored_ssim = ssim(restored_01, x0_01, data_range=1.0).item()
                
                # LPIPS
                y_lpips = lpips_model(y_01 * 2 - 1, x0_01 * 2 - 1).mean().item()
                restored_lpips = lpips_model(restored_01 * 2 - 1, x0_01 * 2 - 1).mean().item()
                
                # 儲存結果
                results[q]['psnr'].append(restored_psnr - y_psnr)  # PSNR增益
                results[q]['ssim'].append(restored_ssim - y_ssim)  # SSIM增益
                results[q]['lpips'].append(y_lpips - restored_lpips)  # LPIPS減少量
                
                # 定期保存一些視覺化結果
                if idx < 10:
                    os.makedirs(f"./avif_test_results/quality_{q}", exist_ok=True)
                    
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(x0[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title("Original")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(y[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title(f"AVIF Q{q}\nPSNR: {y_psnr:.2f}dB\nSSIM: {y_ssim:.4f}")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(restored[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title(f"Restored\nPSNR: {restored_psnr:.2f}dB\nSSIM: {restored_ssim:.4f}")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'./avif_test_results/quality_{q}/sample_{idx+1}.png')
                    plt.close()
        
        # 報告平均結果
        print("\n====== AVIF Average Improvement ======")
        for q in quality_levels:
            avg_psnr_gain = sum(results[q]['psnr']) / len(results[q]['psnr'])
            avg_ssim_gain = sum(results[q]['ssim']) / len(results[q]['ssim'])
            avg_lpips_gain = sum(results[q]['lpips']) / len(results[q]['lpips'])
            print(f"Quality {q}: PSNR Gain = {avg_psnr_gain:.2f}dB, SSIM Gain = {avg_ssim_gain:.4f}, LPIPS Improvement = {avg_lpips_gain:.4f}")

# 主訓練函數
def train_model_ddrm_avif(epochs=100):
    model = AVIFDiffusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    
    best_val_psnr = 0
    train_losses = []
    val_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    for epoch in range(epochs):
        # 訓練一個周期
        loss = train_epoch_ddrm_avif(model, train_dataloader, epoch, optimizer, scheduler)
        train_losses.append(loss)
        
        # 在小集合上驗證
        val_psnr, val_ssim, val_lpips = validate_ddrm_avif(model, valid_dataloader, epoch)
        val_metrics['psnr'].append(val_psnr)
        val_metrics['ssim'].append(val_ssim)
        val_metrics['lpips'].append(val_lpips)
        
        # 保存最佳模型
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_lpips': val_lpips
            }, 'best_ddrm_avif_model.pth')
            print(f"保存新的最佳AVIF模型，PSNR {val_psnr:.2f}dB，SSIM {val_ssim:.4f}，LPIPS {val_lpips:.4f}")
        
        # 繪制訓練曲線
        plot_training_curves_avif(train_losses, val_metrics, epoch)
        
        # 定期顯示還原樣本
        if epoch % 5 == 0 or epoch == epochs - 1:
            visualize_avif_restoration(model, epoch)
    
    print("AVIF DDPM訓練完成！")
    
    # 加載最佳模型並評估
    checkpoint = torch.load('best_ddrm_avif_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加載來自epoch {checkpoint['epoch']+1}的最佳AVIF模型")
    
    # 在不同質量級別上測試
    test_avif_restoration(model, quality_levels=[10, 20, 50, 80])

# 繪製訓練曲線
def plot_training_curves_avif(train_losses, val_metrics, epoch):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AVIF Training Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_metrics['psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('AVIF Validation PSNR')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_metrics['ssim'], label='SSIM')
    plt.plot(val_metrics['lpips'], label='LPIPS')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('AVIF SSIM and LPIPS')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("./avif_curves", exist_ok=True)
    plt.savefig(f'./avif_curves/avif_training_curves_epoch_{epoch}.png')
    plt.close()

# 擴散模型超參數
steps = 100

# 執行訓練
if __name__ == "__main__":
    # 創建必要的目錄
    os.makedirs("./avif_viz", exist_ok=True)
    os.makedirs("./avif_test_results", exist_ok=True)
    os.makedirs("./avif_curves", exist_ok=True)
    
    # 開始AVIF DDPM訓練
    train_model_ddrm_avif(epochs=100)
