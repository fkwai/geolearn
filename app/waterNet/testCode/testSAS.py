import torch
import numpy as np
import scipy.special as sp
from torch.nn import Parameter

nt = 10

x = torch.tensor(0.5)

a = torch.tensor(2)
b = torch.tensor(5)

def betaPdf(x, a, b):
    return torch.exp(torch.lgamma(a + b) - torch.lgamma(b) - torch.lgamma(a) + (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x))

