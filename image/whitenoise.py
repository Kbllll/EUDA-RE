import numpy as np


def add_white_noise(x, sigma=0.01, clip=True):
    """
    对单张图像添加白噪声

    Args:
        x: (H, W) 或 (H, W, C) 图像，值域 [0, 1]
        sigma: 噪声标准差（建议 0.01~0.1）
        clip: 是否将结果裁剪到 [0, 1]

    Returns:
        带噪声的图像（同 shape）
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    x_noisy = x + noise
    if clip:
        x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return x_noisy


def augment_with_white_noise(X, y, n_aug_per_sample=1, sigma=0.01):
    """
    对整个数据集进行白噪声增强

    Args:
        X: (N, H*W) 原始数据，[0,1] 归一化
        y: (N,) 标签
        n_aug_per_sample: 每个原始样本生成多少个带噪版本
        sigma: 噪声强度

    Returns:
        X_aug: (N * (1 + n_aug_per_sample), H*W)
        y_aug: (N * (1 + n_aug_per_sample),)
    """
    img_size = int(np.sqrt(X.shape[1]))  # 假设正方形图像
    X_imgs = X.reshape(-1, img_size, img_size)

    X_aug_list = [X]  # 保留原始数据
    y_aug_list = [y]

    for _ in range(n_aug_per_sample):
        X_noisy = []
        for x in X_imgs:
            x_noisy = add_white_noise(x, sigma=sigma)
            X_noisy.append(x_noisy.flatten())
        X_aug_list.append(np.array(X_noisy))
        y_aug_list.append(y)  # 标签不变

    return np.vstack(X_aug_list), np.hstack(y_aug_list)