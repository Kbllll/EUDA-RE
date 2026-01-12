import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math


# ========================
# 1. 扩散模型（全连接版，适配小图像）
# ========================
class SinusoidalPositionEmbeddings(nn.Module):
    """时间步 t 的位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionModel(nn.Module):
    def __init__(self, img_size, time_emb_dim=32):
        super().__init__()
        self.img_size = img_size
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # 全连接 U-Net（简化）
        self.fc1 = nn.Linear(img_size * img_size, 256)
        self.fc2 = nn.Linear(256 + time_emb_dim, 256)
        self.fc3 = nn.Linear(256, img_size * img_size)
        self.act = nn.ReLU()

    def forward(self, x, t):
        # x: (B, H*W), t: (B,)
        x = self.act(self.fc1(x))
        t_emb = self.time_mlp(t)
        x = torch.cat([x, t_emb], dim=-1)
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


# ========================
# 2. 扩散过程工具
# ========================
class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cpu"):
        self.T = T
        self.img_size = img_size
        self.device = device

        # 线性 beta schedule
        self.beta = torch.linspace(beta_start, beta_end, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x_0, t):
        """前向加噪: q(x_t | x_0)"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1)
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def sample(self, model, n_samples, guidance_scale=0.0):
        """反向采样生成新样本"""
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.img_size * self.img_size).to(self.device)
            for i in reversed(range(self.T)):
                t = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
                noise_pred = model(x, t)
                # 无分类器引导（guidance_scale=0）
                alpha = self.alpha[t].view(-1, 1)
                alpha_bar = self.alpha_bar[t].view(-1, 1)
                beta = self.beta[t].view(-1, 1)

                if i == 0:
                    noise = 0
                else:
                    noise = torch.randn_like(x)

                x = (1 / torch.sqrt(alpha)) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * noise_pred
                ) + torch.sqrt(beta) * noise
        return x


# ========================
# 3. 训练单个类别的扩散模型
# ========================
def train_diffusion_for_class(X_class, img_size, T=200, epochs=20, batch_size=32, device="cpu"):
    """训练单个类别的扩散模型"""
    # 转为 [-1, 1]
    X_norm = (X_class * 2.0) - 1.0
    dataset = DataLoader(
        TensorDataset(torch.tensor(X_norm, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True
    )

    model = DiffusionModel(img_size).to(device)
    diffusion = Diffusion(T=T, img_size=img_size, device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        for (x_0,) in dataset:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            t = torch.randint(0, T, (batch_size,), device=device).long()
            x_t, noise = diffusion.add_noise(x_0, t)
            noise_pred = model(x_t, t)
            loss = nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    return model, diffusion


# ========================
# 4. 主函数：生成带有效标签的增强数据
# ========================
def augment_with_diffusion(X, y, img_size, device="cpu"):
    """
    使用扩散模型生成带有效标签的增强数据
    """
    unique_classes = np.unique(y)
    X_list, y_list = [], []

    for cls in unique_classes:
        idx = (y == cls)
        X_cls = X[idx]
        # print(f"训练扩散模型 for class {cls} (样本数: {len(X_cls)})")


        model, diffusion = train_diffusion_for_class(
            X_cls, img_size, T=200, epochs=15, device=device
        )

        # 生成新样本
        fake_imgs = diffusion.sample(model, len(X_cls))
        fake_imgs = (fake_imgs.cpu().numpy() + 1.0) / 2.0  # 转回 [0,1]
        X_fake = np.clip(fake_imgs, 0, 1)  # 确保在 [0,1]

        X_list.append(X_fake)
        y_list.append(np.full(len(X_cls), cls))

    return np.vstack(X_list), np.hstack(y_list)
