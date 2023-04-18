import torch
import torch.nn.functional as F
import numpy as np

nh = 3
ns = 4
nr = 5
nt = nr
x = torch.randn(nt, ns, nh)  # 4 days, 2 basins, 3 buckets
w = torch.sigmoid(torch.randn(ns, nh * (nr + 1)))
u = torch.relu(w[:, :nh])
r = torch.cumsum(torch.relu(w[:, nh:].view(ns, nh, nr)), dim=-1)
k = torch.softmax(r - u[:, :, None] * nr, dim=-1).permute(2, 0, 1)
q = torch.sum(x * k, dim=0)

# numpy - did not double check
x1 = x.numpy()
w1 = w.numpy()
y1 = np.zeros([nt - nr + 1, ns, nh])
for iS in range(ns):
    for iH in range(nh):
        xx = x[:, iS, iH]
        rr = w1[iS, iH * nr : (iH + 1) * nr]
        y1[:, iS, iH] = np.convolve(xx, np.flip(rr), mode='valid')

torch.mean(y - y1)
