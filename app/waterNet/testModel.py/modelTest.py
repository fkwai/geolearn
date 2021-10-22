import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter as P
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.model import waterNet
import importlib

importlib.reload(waterNet)
P = torch.tensor([1, 2, 3])[:, None]
E = torch.tensor([1, 2, 3])[:, None]
Q = torch.tensor([1, 2, 3])[:, None]/10
T = F.relu(torch.rand(3))[:, None]

x = F.relu(torch.rand(3, 5))
h = F.relu(torch.rand(3, 5))


we = torch.FloatTensor(5).uniform_(-1, 1)
wl = torch.FloatTensor(5).uniform_(-1, 1)
wk = torch.FloatTensor(5).uniform_(-1, 1)
ws = torch.FloatTensor(5).uniform_(-1, 1)

F.softmax(torch.FloatTensor(3, 5), dim=0)

hn = h+x
h1 = torch.relu(hn-torch.exp(wl*2))
q1 = torch.relu(h1-E*torch.sigmoid(we))
h2 = hn-h1
q2 = h2 * torch.sigmoid(wk)
h = h2-q2
q2a = q2*torch.sigmoid(ws)
q2b = q2*(1-torch.sigmoid(ws))

t1 = -8
t2 = 0
x = np.linspace(0, 1, 1000)
y = (t2-t1)/2*np.sin(x*2*np.pi)+(t2+t1)/2
fig, ax = plt.subplots(1, 1)
ax.plot(x, y)
fig.show()

a = np.sum(y > 0)
b = np.sum(y <= 0)
a/(b+a)
1-np.arccos((t2+t1)/(t2-t1))/np.pi

