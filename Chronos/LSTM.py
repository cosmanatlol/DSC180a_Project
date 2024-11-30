import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class Preprocess(Dataset):
    def __init__(self, data, time_steps=10):
        self.scaler = StandardScaler()
        self.scaler.fit(data.reshape(-1, 1))
        self.pred = torch.FloatTensor(self.scaler.transform(data[-time_steps:].reshape(-1, 1)))
        #only outlier transform on pct change data
        data = self.outlier_replace(data, min = -0.15, max = 0.20)
        rft = np.fft.rfft(data)
        rft[15:] = 0
        smooth_data = np.fft.irfft(rft)
        smooth_data[:3], smooth_data[-3:] = data[:3], data[-3:]
        data = self.scaler.transform(smooth_data.reshape(-1, 1))
        self.data, self.target = self.create_sequences(torch.FloatTensor(data), time_steps=time_steps)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def create_sequences(self, data, time_steps):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i])
        return torch.stack(X), torch.stack(y)
    
    def outlier_replace(self, data, min = -0.1, max = 0.2):
        data = np.where(data < min, min, data)
        data = np.where(data > max, max, data)
        return data

    def inv_transform(self, pred):
        return self.scaler.inverse_transform(pred)

    def forecast_element(self):
        return self.pred


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer=50, num_layers=2):
        super(LSTM, self).__init__()
        self.hidden_layer = hidden_layer
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer, num_layers=self.num_layers, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.l1 = nn.Linear(hidden_layer, hidden_layer // 2)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_layer // 2 , output_size)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer).to(input_seq.device)
        out, (hn, _) = self.lstm(input_seq, (h0, c0))
        x = out[:, -1, :]
        x = self.dropout(x)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


def Mase(actual, predicted, roll, test_length):
    naive_forecast = actual[-test_length - roll: -roll]
    actual = actual[-test_length:]
    return mean_absolute_error(actual, predicted) / mean_absolute_error(actual, naive_forecast), mean_absolute_percentage_error(actual, predicted)