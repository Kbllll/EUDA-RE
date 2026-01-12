import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm



from dataload import load_dataset


# ========================
# CNN 模型定义
# ========================
class CNNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNClassifier, self).__init__()
        H, W, C = input_shape

        # 卷积层
        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层（独立定义）
        self.pool1 = nn.MaxPool2d(2)  # 第一块后
        self.pool2 = nn.MaxPool2d(2)  # 第二块后
        self.pool3 = nn.MaxPool2d(2)  # 第三块后（条件使用）

        self.use_pool3 = H >= 32  # 仅 32x32 图像使用第三次池化

        # 计算展平维度（通过前向 dummy pass）
        with torch.no_grad():
            x = torch.zeros(1, C, H, W)
            x = torch.relu(self.conv1(x))
            x = self.pool1(x)
            x = torch.relu(self.conv2(x))
            x = self.pool2(x)
            x = torch.relu(self.conv3(x))
            if self.use_pool3:
                x = self.pool3(x)
            self.flattened_size = x.view(1, -1).size(1)

        # 全连接层
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        if self.use_pool3:
            x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ========================
# 训练函数
# ========================
def train_cnn(
        X=None,
        y=None,
        img_shape=None,
        num_classes=None,
        test_size=0.2,
        random_state=42,
        epochs=20,
        batch_size=32,
        lr=0.001,
        device=None
):
    """
    使用 PyTorch 训练 CNN 分类器

    Args:
        dataset_name: "Digits", "ORL32", "Yale32"
        device: 'cpu' 或 'cuda'（默认自动检测）
    """

    # 自动选择设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    print(f"  图像尺寸: {img_shape}, 类别数: {num_classes}")

    # 转为 PyTorch 张量 (N, C, H, W)
    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1, img_shape[0], img_shape[1])
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = CNNClassifier(input_shape=(img_shape[0], img_shape[1], 1), num_classes=num_classes)
    model.to(device)

    # 优化器 & 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    # 训练循环
    bar = tqdm(range(epochs))
    train_losses, train_accs = [], []
    for _ in bar:
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        bar.set_postfix({
            "Loss": f"{train_losses[-1]:.4f}",
            "Train Acc": f"{train_acc:.2f}%"
        })


    # 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total

    print(f"\n✅ {dataset_name} 最终测试准确率: {test_acc:.2f}%")
    return model, device


