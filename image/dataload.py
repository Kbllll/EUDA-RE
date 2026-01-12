import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from PIL import Image

# 定义目录
DATASETS_DIR = os.path.join(os.getcwd(), "datasets")
CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_Digits():
    cache_file = os.path.join(CACHE_DIR, "digits.npz")
    if os.path.exists(cache_file):
        print("✅ 从缓存加载 Digits")
        data = np.load(cache_file)
        return data["X"], data["y"], data["img_shape"], data["num_classes"]

    """加载 Digits 并归一化到 [0, 1]"""
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target.astype(int)

    n_samples, n_features = X.shape
    # 推断图像尺寸
    if n_features == 64:
        img_shape = (8, 8)
    elif n_features == 1024:
        img_shape = (32, 32)
    else:
        raise ValueError(f"不支持的特征维度: {n_features}")
    num_classes = len(np.unique(y))

    np.savez_compressed(cache_file, X=X, y=y, img_shape=img_shape, num_classes=num_classes)
    print(f"✅ Digits 预处理完成，缓存已保存: {cache_file}")

    return X, y, img_shape, num_classes


def load_ORL32():
    """加载 ORL 人脸数据集（40 人 × 10 张 = 400 张），调整为 32x32"""
    cache_file = os.path.join(CACHE_DIR, "orl32.npz")
    if os.path.exists(cache_file):
        print("✅ 从缓存加载 ORL32")
        data = np.load(cache_file)
        return data["X"], data["y"], data["img_shape"], data["num_classes"]

    base_dir = os.path.join(DATASETS_DIR, "orl_faces")
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"❌ 未找到 ORL 数据集！\n"
            f"请下载并解压到: {base_dir}\n"
            f"下载地址: https://www.kaggle.com/datasets/anaselmasry/orl-face-database"
        )

    images, labels = [], []
    for folder in sorted(os.listdir(base_dir)):
        if not folder.startswith('s') or not folder[1:].isdigit():
            continue
        label = int(folder[1:]) - 1  # s1 → 0, s2 → 1, ...
        folder_path = os.path.join(base_dir, folder)
        for img_file in sorted(os.listdir(folder_path)):
            if img_file.endswith('.pgm'):
                img_path = os.path.join(folder_path, img_file)
                try:
                    with Image.open(img_path).convert('L') as img:
                        img = img.resize((32, 32))
                        arr = np.array(img, dtype=np.float32) / 255.0
                        images.append(arr.flatten())
                        labels.append(label)
                except Exception as e:
                    print(f"⚠️ 跳过损坏图像: {img_path}, error: {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=int)

    n_samples, n_features = X.shape
    # 推断图像尺寸
    if n_features == 64:
        img_shape = (8, 8)
    elif n_features == 1024:
        img_shape = (32, 32)
    else:
        raise ValueError(f"不支持的特征维度: {n_features}")
    num_classes = len(np.unique(y))

    np.savez_compressed(cache_file, X=X, y=y, img_shape=img_shape, num_classes=num_classes)
    print(f"✅ ORL32 预处理完成，缓存已保存: {cache_file}")
    return X, y, img_shape, num_classes


def load_dataset(name):
    """
    统一加载接口
    支持: 'Digits', 'ORL32'
    返回: (X, y) 其中 X 是 (n_samples, n_features) 浮点数组，y 是整数标签
    """
    if name == "Digits":
        return load_Digits()
    elif name == "ORL32":
        return load_ORL32()
    else:
        raise ValueError("不支持的数据集。支持: 'Digits', 'ORL32'")


class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = [x]  # 初始化时将输入的x作为第一个元素
        self.y = [y]  # 初始化时将输入的y作为第一个元素
        self.indices = None
        self.dataset = None

    def __len__(self):
        # 计算所有批次数据的总长度
        return sum(len(item) for item in self.x)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self.get_one_item(idx[i]) for i in range(len(idx))]
        return self.get_one_item(idx)

    def get_one_item(self, idx):
        # 遍历所有批次，找到索引对应的具体数据
        cumulative_length = 0
        for i in range(len(self.x)):
            batch_length = len(self.x[i])
            if idx < cumulative_length + batch_length:
                # 计算在当前批次中的相对索引
                local_idx = idx - cumulative_length
                return self.x[i][local_idx], self.y[i][local_idx]
            cumulative_length += batch_length

        # 如果索引超出范围，抛出异常
        raise IndexError("Index out of range")

    def append(self, x, y):
        # 确保x和y长度一致
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        self.x.append(x)
        self.y.append(y)

    def get_raw(self):
        return self.x[0], self.y[0]

    def get_merged_lists(self):
        """
        合并并返回x和y内部的所有list
        """
        merged_x = []
        merged_y = []
        for sublist in self.x:
            merged_x.extend(sublist)
        for sublist in self.y:
            merged_y.extend(sublist)
        return np.vstack(merged_x), np.array(merged_y)

    def link(self, indices, parent):
        self.indices = indices
        self.dataset = parent

    def sub2init(self, indices):
        x = self.x[0][indices]
        y = self.y[0][indices]
        subset = MyDataSet(x, y)
        subset.link(indices, self)
        return subset

