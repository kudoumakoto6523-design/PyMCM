import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pymcm.core.base import BaseModel


# --- 内部 PyTorch 模型定义 ---
class _LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(_LSTMModule, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        # batch_first=True 意味着输入格式为 (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层 (把 LSTM 的输出映射到最终预测值)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态 h0 和 细胞状态 c0
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        # out 形状: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # 我们只需要最后一个时间步的输出
        out = out[:, -1, :]

        # 经过全连接层
        out = self.fc(out)
        return out


# --- PyMCM 封装类 ---
class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) Predictor.

    Great for non-linear time series with long dependencies.
    """

    def __init__(self, window_size=5, hidden_size=50, num_layers=1, epochs=100, lr=0.01):
        """
        Args:
            window_size (int): Look-back period (how many past steps to predict next).
            hidden_size (int): Number of neurons in LSTM layer.
            num_layers (int): Number of stacked LSTM layers.
            epochs (int): Training iterations.
            lr (float): Learning rate.
        """
        super().__init__(name="LSTM Neural Network")
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 数据归一化很重要
        self.train_data_normalized = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, data):
        """
        Convert 1D array into sequences:
        X: [[t1, t2, t3], [t2, t3, t4]...]
        y: [t4, t5...]
        """
        xs, ys = [], []
        for i in range(len(data) - self.window_size):
            x = data[i:(i + self.window_size)]
            y = data[i + self.window_size]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fit(self, mcm_data):
        # 1. 准备数据
        raw_y = mcm_data.get_y()
        if raw_y is None:  # 容错
            raw_y = mcm_data.get_X()[:, 0]

        # 必须转成 2D 数组 (N, 1) 给 scaler 用
        raw_y = np.array(raw_y).reshape(-1, 1)
        self.raw_data = raw_y  # 存一份原始的

        # 2. 归一化 (LSTM 对数据范围非常敏感，不归一化很难收敛)
        self.train_data_normalized = self.scaler.fit_transform(raw_y)

        # 3. 制作数据集 (Sliding Window)
        X, y = self._create_sequences(self.train_data_normalized)

        # 转成 Tensor
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)

        # X_train 形状必须是 (Batch, Seq_Len, Features)
        # 我们的 Features 是 1 (单变量预测)
        # 所以 reshape 成 (N, window_size, 1)
        X_train = X_train.view(-1, self.window_size, 1)

        # 4. 初始化模型
        self.model = _LSTMModule(input_size=1,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 output_size=1).to(self.device)

        # 5. 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 6. 训练循环
        print(f"[{self.name}] Training on {self.device}...")
        for i in range(self.epochs):
            optimizer.zero_grad()

            y_pred = self.model(X_train)
            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}/{self.epochs}, Loss: {loss.item():.6f}")

        self.is_fitted = True
        return self

    def predict(self, steps=10):
        """
        Autoregressive Prediction (滚动预测).
        Use predicted value as input for next prediction.
        """
        if not self.is_fitted:
            raise Exception("Model not fitted.")

        self.model.eval()  # 切换到评估模式

        # 1. 拿到最后一段已知数据作为起始输入
        # 取训练数据最后 window_size 个点
        current_seq = self.train_data_normalized[-self.window_size:]
        current_seq = torch.FloatTensor(current_seq).to(self.device)
        # reshape: (1, window_size, 1) -> Batch=1
        current_seq = current_seq.view(1, self.window_size, 1)

        predictions = []

        # 2. 循环预测
        with torch.no_grad():
            for _ in range(steps):
                # 预测下一步
                y_next = self.model(current_seq)  # shape (1, 1)
                predictions.append(y_next.item())

                # 更新输入序列：
                # 扔掉最旧的一个，把新预测的加到尾部
                # current_seq shape: (1, window_size, 1)
                # y_next shape: (1, 1) -> view -> (1, 1, 1)
                next_input = y_next.view(1, 1, 1)
                current_seq = torch.cat((current_seq[:, 1:, :], next_input), axis=1)

        # 3. 反归一化 (Inverse Transform)
        # 变回真实的数值范围
        predictions = np.array(predictions).reshape(-1, 1)
        true_predictions = self.scaler.inverse_transform(predictions)

        return true_predictions.flatten()

    def _predict_single(self, x):
        return 0

    def plot(self, steps=10):
        if not self.is_fitted: return

        # 预测未来
        future_pred = self.predict(steps)

        # 拼贴历史数据以便画图
        history = self.raw_data.flatten()

        plt.figure(figsize=(12, 6))

        # 画历史
        plt.plot(np.arange(len(history)), history, label='History', color='black')

        # 画未来 (连接历史的最后一个点，让图好看点)
        future_x = np.arange(len(history) - 1, len(history) + steps)
        # 起点是历史最后一点，后面是预测值
        plot_y = np.concatenate(([history[-1]], future_pred))

        plt.plot(future_x, plot_y, label='LSTM Forecast', color='red', linestyle='--')

        plt.title(f"LSTM Prediction (Window={self.window_size}, Epochs={self.epochs})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()