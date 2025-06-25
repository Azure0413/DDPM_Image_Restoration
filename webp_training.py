import torch
import torchvision
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

# 設置資料載入器
batch_size = 18  # 減小批次大小以應對更大的圖像
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 定義WebP壓縮函數
def webp_compress(x, quality):
    """執行WebP壓縮並返回解碼結果"""
    # 從[-1,1]轉換為[0,255] uint8
    x = (x * 127.5 + 127.5).clamp(0, 255).to(torch.uint8).cpu()
    
    compressed_images = []
    for img in x:
        # 轉換為PIL圖像
        pil_img = torchvision.transforms.ToPILImage()(img)
        
        # 壓縮為WebP
        buffer = io.BytesIO()
        quality = max(0, min(100, int(quality)))  # WebP質量為0-100
        pil_img.save(buffer, format="WEBP", quality=quality)
        buffer.seek(0)
        
        # 解碼WebP
        compressed_img = Image.open(buffer)
        compressed_tensor = torchvision.transforms.ToTensor()(compressed_img)
        compressed_images.append(compressed_tensor)
    
    # 轉換回[-1,1]範圍並返回到設備
    return torch.stack(compressed_images).to(device).sub(0.5).mul(2.0)

# 定義色彩保持和頻率領域感知損失
def frequency_aware_loss(pred, target):
    """結合傳統MSE和頻率域MSE的損失函數"""
    # 空間域MSE
    spatial_loss = F.mse_loss(pred, target)
    
    # 轉換到[0,1]範圍進行計算
    pred_01 = pred * 0.5 + 0.5
    target_01 = target * 0.5 + 0.5
    
    # 頻率域損失 - 對每個通道分別計算DCT變換
    freq_loss = 0
    for c in range(3):
        # 計算DCT系數
        pred_dct = torch.fft.rfft2(pred_01[:, c])
        target_dct = torch.fft.rfft2(target_01[:, c])
        
        # 頻率域的MSE
        freq_mse = F.mse_loss(torch.abs(pred_dct), torch.abs(target_dct))
        # 相位損失
        phase_loss = F.mse_loss(torch.angle(pred_dct), torch.angle(target_dct))
        
        freq_loss += freq_mse + 0.5 * phase_loss
    
    # SSIM感知損失
    ssim_loss = 1.0 - ssim(pred_01, target_01, data_range=1.0, size_average=True)
    
    # 結合損失 - 給高頻信息更大權重
    return spatial_loss + 0.5 * freq_loss + 0.3 * ssim_loss

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

# DCT變換層
class DCTLayer(nn.Module):
    """實現精確的DCT變換操作"""
    def __init__(self, block_size=4):  # WebP使用4x4子塊
        super().__init__()
        self.block_size = block_size
        self.register_buffer('dct_matrix', self._get_dct_matrix(block_size))
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 填充至block_size的整數倍
        h_pad = (self.block_size - h % self.block_size) % self.block_size
        w_pad = (self.block_size - w % self.block_size) % self.block_size
        
        x_padded = F.pad(x, (0, w_pad, 0, h_pad))
        
        # 計算填充後的總高度和寬度
        h_padded = h + h_pad
        w_padded = w + w_pad
        
        # 分割圖像成塊
        patches = x_padded.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        patches = patches.contiguous().view(-1, self.block_size, self.block_size)
        
        # 執行DCT: D * X * D^T
        dct_coeffs = torch.matmul(torch.matmul(self.dct_matrix, patches), self.dct_matrix.transpose(0, 1))
        
        # 重構回原始形狀
        dct_blocks = dct_coeffs.view(b, c, h_padded // self.block_size, w_padded // self.block_size, 
                                    self.block_size, self.block_size)
        # 排列回空間域順序
        dct_spatial = dct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        dct_spatial = dct_spatial.view(b, c, h_padded, w_padded)
        
        # 移除填充
        if h_pad > 0 or w_pad > 0:
            dct_spatial = dct_spatial[:, :, :h, :w]
            
        return dct_spatial
    
    def _get_dct_matrix(self, size):
        """生成標準離散餘弦變換矩陣"""
        dct_matrix = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / torch.sqrt(torch.tensor(size, dtype=torch.float32))
                else:
                    dct_matrix[i, j] = torch.sqrt(torch.tensor(2.0 / size)) * torch.cos(torch.tensor(torch.pi * (2 * j + 1) * i / (2 * size)))
        return dct_matrix

# WebP頻率感知塊
class WebPFreqAwareBlock(nn.Module):
    """特別設計用於處理WebP壓縮的頻率感知模塊"""
    def __init__(self, channels, block_size=4):  # 使用4×4塊，如VP8
        super().__init__()
        self.block_size = block_size
        self.dct = DCTLayer(block_size)
        
        # 頻率注意力 - 針對WebP的算術編碼調整
        self.low_freq_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        self.high_freq_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        
        # 輸出層
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x, compression_level=None):
        # DCT頻率表示
        x_dct = self.dct(x)
        
        # 分離低頻和高頻
        b, c, h, w = x_dct.shape
        low_freq = torch.zeros_like(x_dct)
        high_freq = torch.zeros_like(x_dct)
        
        # 按塊處理頻率
        for i in range(0, h, self.block_size):
            i_end = min(i + self.block_size, h)
            for j in range(0, w, self.block_size):
                j_end = min(j + self.block_size, w)
                
                # WebP對低頻處理與JPEG不同
                low_size = max(1, min(3, min(i_end - i, j_end - j)))
                low_freq[:, :, i:i+low_size, j:j+low_size] = x_dct[:, :, i:i+low_size, j:j+low_size]
                
                # 高頻部分
                high_freq[:, :, i:i_end, j:j_end] = x_dct[:, :, i:i_end, j:j_end]
                high_freq[:, :, i:i+low_size, j:j+low_size] = 0
        
        # 應用注意力
        low_attn = self.low_freq_attn(low_freq)
        high_attn = self.high_freq_attn(high_freq)
        
        # WebP一般比JPEG更好地保留高頻
        if compression_level is not None:
            if isinstance(compression_level, torch.Tensor) and compression_level.dim() > 0:
                compression_level = compression_level.view(-1, 1, 1, 1)
            # 針對WebP特性調整
            high_boost = torch.clamp(1.0 - compression_level, 0.15, 1.9)
            high_attn = high_attn * high_boost
        
        # 組合注意力結果
        combined = low_attn * low_freq + high_attn * high_freq
        
        # 轉回空間域並添加殘差連接
        return self.conv_out(x + combined)

# 改進的殘差注意力塊，整合WebP頻率感知
class WebPResAttnBlock(nn.Module):
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
        
        # 自注意力機制
        self.attn = nn.MultiheadAttention(out_c, 4, batch_first=True)
        
        # WebP特定頻率處理
        self.freq_guide = WebPFreqAwareBlock(out_c)
        
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
        
        # 應用WebP特定頻率處理
        h = self.freq_guide(h, compression_level)
        
        # 殘差連接
        return self.shortcut(x) + h

# 完整的UNet架構，專為WebP偽影去除設計
class WebPDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_embed = TimeEmbedding(time_dim)
        
        # 下採樣路徑
        self.down1 = WebPResAttnBlock(3, 64, time_dim)
        self.down2 = WebPResAttnBlock(64, 128, time_dim)
        self.down3 = WebPResAttnBlock(128, 256, time_dim)
        self.down4 = WebPResAttnBlock(256, 512, time_dim)
        self.down5 = WebPResAttnBlock(512, 512, time_dim)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶頸層
        self.bottleneck = nn.Sequential(
            WebPResAttnBlock(512, 1024, time_dim),
            WebPResAttnBlock(1024, 1024, time_dim),
            WebPResAttnBlock(1024, 512, time_dim)
        )
        
        # 上採樣路徑
        self.up1 = WebPResAttnBlock(1024, 512, time_dim)
        self.up2 = WebPResAttnBlock(1024, 256, time_dim)
        self.up3 = WebPResAttnBlock(512, 128, time_dim)
        self.up4 = WebPResAttnBlock(256, 64, time_dim)
        self.up5 = WebPResAttnBlock(128, 64, time_dim)
        
        # DCT感知層
        self.dct_layer = DCTLayer(block_size=4)  # VP8使用4×4子塊
        
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
        
        # 應用DCT層增強頻率感知
        dct_feature = self.dct_layer(u5)
        combined = u5 + 0.1 * dct_feature  # 輕微融合DCT特徵
        
        return self.out_conv(combined)

# 相位一致性函數 - 保持圖像結構特徵
def phase_consistency(x, ref, alpha=0.7):
    """使用傅里葉變換的相位一致性，保持頻域特性"""
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

# DDRM-WebP採樣器
class DDRMWebPSampler:
    def __init__(self, model):
        self.model = model
        
    def sample(self, x_t, quality, steps=100, eta=0.85, eta_b=1.0):
        """DDRM-WebP採樣方法，專為WebP偽影去除設計"""
        self.model.eval()
        
        # 保存原始壓縮圖像作為測量值y
        y = x_t.clone()
        
        with torch.no_grad():
            # 反向擴散過程
            for i in tqdm(range(steps-1, -1, -1), desc="Sampling"):
                # 計算標準化時間步
                t = torch.full((x_t.size(0),), i, device=device).float() / steps
                
                # 下一個時間步（用於噪聲縮放）
                t_next = torch.full((x_t.size(0),), max(0, i-1), device=device).float() / steps
                
                # 壓縮級別與時間步關聯
                compression_level = t.clone()
                
                # 模型預測
                x_theta = self.model(x_t, t, compression_level)
                
                # DDRM-WebP更新規則
                # 首先，對預測結果進行WebP壓縮
                webp_x_theta = webp_compress(x_theta, quality)
                
                # 根據DDRM公式計算校正項
                x_prime = x_theta - webp_x_theta + y
                
                if i > 0:
                    # 計算噪聲
                    noise_scale = t.float() * 0.2
                    random_noise = torch.randn_like(x_t) * noise_scale.view(-1, 1, 1, 1)
                    
                    # 混合校正項、預測和噪聲
                    x_t = eta_b * x_prime + (1 - eta_b) * x_theta + eta * random_noise
                    
                    # 低質量WebP的額外穩定處理
                    if quality < 15 and i % 5 == 0:  # WebP調整閥值
                        # 應用相位一致性以保留邊緣
                        x_t = phase_consistency(x_t, y, alpha=0.7)
                else:
                    # 最後一步 - 只使用校正後的預測
                    x_t = x_prime
        
        return x_t

# 更新訓練函數
def train_epoch_ddrm_webp(model, loader, epoch, optimizer, scheduler):
    model.train()
    total_loss = 0
    freq_loss_total = 0
    ssim_loss_total = 0
    
    for x0, _ in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        x0 = x0.to(device)
        b = x0.size(0)
        
        # 質量選擇策略 - 自適應增加高質量比例
        epoch_progress = min(1.0, epoch / 100)  # 標準化到[0,1]
        if random.random() < 0.3 + 0.4 * epoch_progress:
            # 高質量
            quality_range = (70, 100)
        elif random.random() < 0.5:
            # 中等質量
            quality_range = (40, 70)
        else:
            # 低質量 - WebP從0開始
            quality_range = (0, 40)
            
        # 隨機時間步選擇
        t = torch.randint(1, steps, (b,), device=device).long()
        
        # 基於時間步計算每個樣本的質量
        min_q, max_q = quality_range
        quality = torch.clamp(min_q + (max_q - min_q) * (1 - t.float() / steps), 0, 100).cpu().numpy()
        
        # 應用WebP壓縮獲取帶噪聲圖像
        xt = torch.stack([webp_compress(x0[i:i+1], int(q)) for i, q in enumerate(quality)])
        if xt.dim() > 4:  # 處理批次維度被擴展的情況
            xt = xt.squeeze(1)
        
        # 計算目標（噪聲/殘差）
        target = x0 - xt
        
        # 獲取模型預測
        compression_level = t.float() / steps
        pred = model(xt, t.float()/steps, compression_level)
        
        # 計算頻率感知損失
        loss = frequency_aware_loss(xt + pred, x0)
        
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
def validate_ddrm_webp(model, loader, epoch):
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
            qualities = [10, 30, 50]
            
            for quality in qualities:
                # 創建壓縮圖像
                y = webp_compress(x0, quality)
                
                # 設置初始時間步與質量相關
                init_t = int((100 - quality) / 100 * steps)
                init_t = max(20, min(init_t, 80))  # 保持合理範圍
                
                # 使用採樣器恢復
                sampler = DDRMWebPSampler(model)
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
        visualize_webp_restoration(model, epoch)
    
    return avg_psnr, avg_ssim, avg_lpips

# 可視化結果函數
def visualize_webp_restoration(model, epoch):
    model.eval()
    sampler = DDRMWebPSampler(model)
    
    with torch.no_grad():
        x0, _ = next(iter(test_dataloader))
        x0 = x0.to(device)
        
        # 測試不同的質量級別
        qualities = [5, 10, 30, 50]
        plt.figure(figsize=(len(qualities)*3+3, 5))
        
        # 顯示原始圖像
        plt.subplot(2, len(qualities)+1, 1)
        plt.imshow(x0[0].cpu().permute(1,2,0)*0.5+0.5)
        plt.title("Original")
        plt.axis('off')
        
        # 對每個質量級別顯示WebP和還原結果
        for i, q in enumerate(qualities):
            # WebP壓縮
            y = webp_compress(x0, q)
            
            # 設定初始時間步長對應質量
            init_t = int((100 - q) / 100 * steps)
            init_t = max(20, min(init_t, 80))  # 保持合理範圍
            
            # 使用採樣器進行還原
            restored = sampler.sample(y, q, steps=init_t)
            
            # 計算PSNR
            x0_01 = (x0 * 0.5 + 0.5).clamp(0, 1)
            y_01 = (y * 0.5 + 0.5).clamp(0, 1)
            restored_01 = (restored * 0.5 + 0.5).clamp(0, 1)
            
            y_psnr = -10 * math.log10(F.mse_loss(y_01, x0_01).item())
            restored_psnr = -10 * math.log10(F.mse_loss(restored_01, x0_01).item())
            
            # 顯示WebP壓縮結果
            plt.subplot(2, len(qualities)+1, i+2)
            plt.imshow(y[0].cpu().permute(1,2,0)*0.5+0.5)
            plt.title(f"WebP Q{q}\nPSNR: {y_psnr:.2f}dB")
            plt.axis('off')
            
            # 顯示還原結果
            plt.subplot(2, len(qualities)+1, len(qualities)+i+2)
            plt.imshow(restored[0].cpu().permute(1,2,0)*0.5+0.5)
            plt.title(f"Restored\nPSNR: {restored_psnr:.2f}dB")
            plt.axis('off')
        
        plt.tight_layout()
        os.makedirs("./viz", exist_ok=True)
        plt.savefig(f'./viz/webp_restoration_epoch_{epoch}.png')
        plt.close()

# 完整測試函數
def test_webp_restoration(model, quality_levels=[5, 10, 30, 50]):
    # 初始化採樣器
    sampler = DDRMWebPSampler(model)
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
                # WebP壓縮
                y = webp_compress(x0, q)
                
                # 設定初始時間步長對應質量
                init_t = int((100 - q) / 100 * steps)
                init_t = max(20, min(init_t, 80))
                
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
                    os.makedirs(f"./test_results/quality_{q}", exist_ok=True)
                    
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(x0[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title("Original")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(y[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title(f"WebP Q{q}\nPSNR: {y_psnr:.2f}dB\nSSIM: {y_ssim:.4f}")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(restored[0].cpu().permute(1,2,0)*0.5+0.5)
                    plt.title(f"Restored\nPSNR: {restored_psnr:.2f}dB\nSSIM: {restored_ssim:.4f}")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'./test_results/quality_{q}/sample_{idx+1}.png')
                    plt.close()
        
        # 報告平均結果
        print("\n====== Average Improvement ======")
        for q in quality_levels:
            avg_psnr_gain = sum(results[q]['psnr']) / len(results[q]['psnr'])
            avg_ssim_gain = sum(results[q]['ssim']) / len(results[q]['ssim'])
            avg_lpips_gain = sum(results[q]['lpips']) / len(results[q]['lpips'])
            print(f"Quality {q}: PSNR Gain = {avg_psnr_gain:.2f}dB, SSIM Gain = {avg_ssim_gain:.4f}, LPIPS Improvement = {avg_lpips_gain:.4f}")

# 繪製訓練曲線
def plot_training_curves(train_losses, val_metrics, epoch):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_metrics['psnr'], label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_metrics['ssim'], label='SSIM')
    plt.plot(val_metrics['lpips'], label='LPIPS')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('SSIM and LPIPS')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("./curves", exist_ok=True)
    plt.savefig(f'./curves/training_curves_epoch_{epoch}.png')
    plt.close()

# 主訓練函數
def train_model_ddrm_webp(epochs=100):
    model = WebPDiffusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    
    best_val_psnr = 0
    train_losses = []
    val_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    
    for epoch in range(epochs):
        # 訓練一個周期
        loss = train_epoch_ddrm_webp(model, train_dataloader, epoch, optimizer, scheduler)
        train_losses.append(loss)
        
        # 在小集合上驗證
        val_psnr, val_ssim, val_lpips = validate_ddrm_webp(model, valid_dataloader, epoch)
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
            }, 'best_ddrm_webp_model.pth')
            print(f"保存新的最佳模型，PSNR {val_psnr:.2f}dB，SSIM {val_ssim:.4f}，LPIPS {val_lpips:.4f}")
        
        # 繪制訓練曲線
        plot_training_curves(train_losses, val_metrics, epoch)
        
        # 定期顯示還原樣本
        if epoch % 5 == 0 or epoch == epochs - 1:
            visualize_webp_restoration(model, epoch)
    
    print("訓練完成！")
    
    # 加載最佳模型並評估
    checkpoint = torch.load('best_ddrm_webp_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加載來自epoch {checkpoint['epoch']+1}的最佳模型")
    
    # 在不同質量級別上測試
    test_webp_restoration(model, quality_levels=[0, 5, 10, 30, 50])

# 擴散模型超參數
steps = 100

# 執行訓練
if __name__ == "__main__":
    # 創建必要的目錄
    os.makedirs("./viz", exist_ok=True)
    os.makedirs("./test_results", exist_ok=True)
    os.makedirs("./curves", exist_ok=True)
    
    # 開始訓練
    train_model_ddrm_webp(epochs=100)