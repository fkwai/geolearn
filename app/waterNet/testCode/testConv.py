import torch
import torch.nn.functional as F
import numpy as np

nt = 5
nh = 3
ns = 4
nr = 5
x = torch.randn(nt, ns, nh)  # 4 days, 2 basins, 3 buckets
w = torch.sigmoid(torch.randn(ns, nh * nr))
r = w.view(ns * nh, 1, nr)
a = x.permute(1, 2, 0).view(1, ns * nh, nt)
y = F.conv1d(a, r, groups=ns * nh).view(ns, nh, nt - nr + 1).permute(2, 0, 1)


r = torch.softmax(w.view(ns, nh, nr), dim=-1)

# numpy
x1 = x.numpy()
w1 = w.numpy()
y1 = np.zeros([nt - nr + 1, ns, nh])
for iS in range(ns):
    for iH in range(nh):
        xx = x[:, iS, iH]
        rr = w1[iS, iH * nr : (iH + 1) * nr]
        y1[:, iS, iH] = np.convolve(xx, np.flip(rr), mode='valid')

torch.mean(y - y1)
