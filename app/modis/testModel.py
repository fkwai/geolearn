
import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


a = np.ones([1, 1, 3])
b = np.random.rand(6)

aa = th.from_numpy(a).float()
conv = nn.Conv1d(1, 1, 3, bias=False)

deconv = nn.ConvTranspose1d(1, 1, 3, bias=False)
deconv.weight = conv.weight

weight = conv.weight.detach().numpy()
bb = conv(aa)
cc = deconv(bb)
