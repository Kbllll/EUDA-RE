import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# ========================
# 1. GAN
# ========================
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, img_size * img_size),
            nn.Tanh()  # 输出 [-1, 1]
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_size * img_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ========================
# 2. 训练单个类别的 GAN
# ========================
def train_gan_for_class(X_class, img_size, latent_dim=100, epochs=15, device=None):
    """训练单个类别的 GAN"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转为 [-1, 1]
    X_gan = (X_class * 2.0) - 1.0
    dataset = TensorDataset(torch.tensor(X_gan, dtype=torch.float32).view(-1, 1, img_size, img_size))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    netG = Generator(latent_dim, img_size).to(device)
    netD = Discriminator(img_size).to(device)

    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for real_img in dataloader:
            real_img = real_img[0].to(device)
            b_size = real_img.size(0)

            # Train Discriminator
            netD.zero_grad()
            label_real = torch.ones(b_size, 1, device=device)
            label_fake = torch.zeros(b_size, 1, device=device)

            lossD_real = criterion(netD(real_img), label_real)
            noise = torch.randn(b_size, latent_dim, device=device)
            fake_img = netG(noise)
            lossD_fake = criterion(netD(fake_img.detach()), label_fake)
            (lossD_real + lossD_fake).backward()
            optD.step()

            # Train Generator
            netG.zero_grad()
            lossG = criterion(netD(fake_img), label_real)
            lossG.backward()
            optG.step()

    return netG


# ========================
# 3. 主函数：生成带有效标签的增强数据
# ========================
def augment_with_valid_labels(X, y, img_size, latent_dim=100, device=None):
    """
    使用 GAN 生成带有效标签的增强数据

    Args:
        X: (N, H*W) 原始数据，[0,1] 归一化
        y: (N,) 真实标签 (0, 1, ..., C-1)
        img_size: 图像边长 (8 or 32)
        augment_per_class: 每类生成多少新样本

    Returns:
        X_aug: (N + C*K, H*W) 增强后数据
        y_aug: (N + C*K,) 对应有效标签
    """
    unique_classes = np.unique(y)
    X_list = []
    y_list = []

    for cls in unique_classes:
        # 获取该类样本
        idx = (y == cls)
        X_cls = X[idx]
        # print(f"训练 GAN for class {cls} (样本数: {len(X_cls)})")

        # 训练 GAN
        generator = train_gan_for_class(X_cls, img_size, latent_dim=latent_dim, device=device)

        # 生成新样本
        augment_per_class = len(X_cls)
        noise = torch.randn(augment_per_class, latent_dim, device=device)
        with torch.no_grad():
            fake_imgs = generator(noise)
        fake_imgs = (fake_imgs.cpu().numpy() + 1.0) / 2.0  # 转回 [0,1]
        X_fake = fake_imgs.reshape(augment_per_class, -1)

        # 添加到列表（标签 = cls）
        X_list.append(X_fake)
        y_list.append(np.full(augment_per_class, cls))

    return np.vstack(X_list), np.hstack(y_list)

