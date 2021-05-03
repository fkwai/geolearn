
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(LSTM, self).__init__()
        self.ct = 0
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.relu = nn.ReLU()
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, 2, dropout=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = self.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0)
        out = self.linearOut(outLSTM)
        return out
