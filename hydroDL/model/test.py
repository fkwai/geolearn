
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import PositionalEncoding
import math


class LSTM(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(LSTM, self).__init__()
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.dropout = nn.Dropout(p=dr)
        self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, 2, dropout=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, doDropMC=False):
        x0 = self.dropout(self.linearIn(x))
        y, (hn, cn) = self.lstm(x0)
        out = self.linearOut(y)
        # out = y.mean(dim=2, keepdim=True)*10
        return out


class LSTM2(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(LSTM2, self).__init__()
        self.linearIn1 = torch.nn.Linear(nx-1, hiddenSize)
        self.linearIn2 = torch.nn.Linear(1, hiddenSize)
        self.dropout = nn.Dropout(p=dr)
        self.lstm1 = torch.nn.LSTM(hiddenSize, hiddenSize, 2, dropout=dr)
        self.lstm2 = torch.nn.LSTM(hiddenSize*2, hiddenSize, 2, dropout=dr)
        self.linearOut1 = torch.nn.Linear(hiddenSize, 1)
        self.linearOut2 = torch.nn.Linear(hiddenSize, ny-1)
        self.gpu = 1

    def forward(self, x, q, doDropMC=False):
        x0 = self.dropout(self.linearIn1(x))
        h1, (hn1, cn1) = self.lstm1(x0)
        x1 = self.dropout(torch.cat((self.linearIn2(q), h1), dim=2))
        h2, (hn2, cn2) = self.lstm2(x1)
        out1 = self.linearOut1(h1)
        x1 = self.dropout(self.linearIn2(q))
        out2 = self.linearOut2(h2)
        return out1, out2


class GRU(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize, dr=0.5):
        super(GRU, self).__init__()
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.dropout = nn.Dropout(p=dr)
        self.gru = torch.nn.LSTM(hiddenSize, hiddenSize, 2, dropout=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x,  doDropMC=False):
        x0 = self.dropout(self.linearIn(x))
        y, hn = self.gru(x0)
        out = self.linearOut(torch.tanh(y))
        # out = y.mean(dim=2, keepdim=True)*10
        return out


class Transformer(torch.nn.Module):
    def __init__(self, nx, ny, hiddenSize=512, nhead=8, nlayer=6, dr=0.5):
        super(Transformer, self).__init__()
        self.d_model = hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.posEncoder = PositionalEncoding(hiddenSize, dr).cuda()

        encoder = nn.TransformerEncoderLayer(
            d_model=hiddenSize, nhead=nhead, dropout=dr, activation='gelu')
        self.transEncoder = nn.TransformerEncoder(encoder, nlayer)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linearIn.bias.data.zero_()
        self.linearIn.weight.data.uniform_(-initrange, initrange)
        self.linearOut.bias.data.zero_()
        self.linearOut.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, doDropMC=False):
        x0 = self.linearIn(x) * math.sqrt(self.d_model)
        x1 = self.posEncoder(x0)
        x2 = self.transEncoder(x1)
        out = self.linearOut(x2)
        return out
