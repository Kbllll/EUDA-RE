import datetime
import os
import sys
import traceback

import numpy as np
from sklearn.model_selection import train_test_split
import uuid

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from dataload import load_dataset, MyDataSet
from enhance import Enhancer
from model import CNNClassifier


class Exp:
    def __init__(self, args):
        self.args = args
        self.device = utils.get_device(args.device)
        self.args.device = str(self.device)
        print(self.device)

        self.enhancer = None
        self.set_enhance()

        self.train_data = None
        self.test_data = None
        self.set_data()

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.set_model()
        utils.print_args(self.args)

    def set_enhance(self):
        self.enhancer = Enhancer(self.args) if self.args.enhance else None

    def set_data(self):
        X, y, self.img_shape, self.args.num_classes = load_dataset(self.args.dataset_name)
        self.args.num_classes = int(torch.tensor(self.args.num_classes).item())  # 转换为整数

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.args.test_size,
            stratify=y,
            random_state=self.args.seed,
        )

        self.train_data = MyDataSet(X_train, y_train)
        self.test_data = MyDataSet(X_test, y_test)



    def set_model(self):
        # 初始化模型
        self.model = CNNClassifier(input_shape=(self.img_shape[0], self.img_shape[1], 1),
                                   num_classes=self.args.num_classes).to(self.device)
        self.criterion = F.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)


    def exp_init(self):
        if self.enhancer:
            x_extra, y_extra = self.enhancer.generate(self.train_data)
            self.train_data.append(x_extra, y_extra)
            pass

    def train(self):

        X_train,y_train = self.train_data.get_merged_lists()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, self.img_shape[0], self.img_shape[1])
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch,
            shuffle=True,
        )

        self.model.train()

        bar = tqdm(range(self.args.epochs))
        train_losses, train_accs = [], []
        for _ in bar:
            train_loss, correct, total = 0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

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

    def test(self):
        X_test,y_test = self.test_data.get_raw()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, self.img_shape[0], self.img_shape[1])
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch,
            shuffle=False
        )

        # 验证
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        self.args.test_acc = 100 * correct / total

        print(f"\n✅ {self.args.dataset_name} 最终测试准确率: {self.args.test_acc:.2f}%")



    def save(self):
        path = './out'
        if not os.path.exists(path):
            os.makedirs(path)
        uid = uuid.uuid4().hex[:8]
        setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
        out_path = os.path.join(path, setting)
        try:
            os.makedirs(out_path)
            utils.save_args_to_json(self.args, os.path.join(out_path, 'args.json'))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
