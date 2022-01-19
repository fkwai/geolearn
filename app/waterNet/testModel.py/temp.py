
import numpy as np
import torch
import torch.nn.functional as F

nt = 100
s = np.zeros(nt)
s0 = 0
g = 0.01
b = 2
p = 1
for k in range(nt):
    s[k] = (1-g)*(s0+p)-b
    s0 = s[k]
h = s+b/g
q = g*(s+p)+b
