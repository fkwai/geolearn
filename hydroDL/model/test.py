
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(LSTM, self).__init__()
        self.relu = nn.ReLU()
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.dropout = nn.Dropout(p=dr)
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, 3, dropout=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = self.dropout(self.relu(self.linearIn(x)))
        y, (hn, cn) = self.lstm(x0)
        out = self.linearOut(y)
        return out


class Transformer(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(Transformer, self).__init__()
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.trans = nn.Transformer(nhead=16, num_encoder_layers=12)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = self.linearIn(x)
        y, (hn, cn) = self.trans(x0)
        out = self.linearOut(y)
        return out
