import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

nt=3
nh=2
x=torch.Tensor([1,2,3])[:,None]
o=torch.Tensor([1,2,3])[:,None]
h0=torch.zeros(nh)
w1=Parameter(torch.Tensor(1, nh))
w2=Parameter(torch.Tensor(nh, 1))

w1.data.uniform_(-1, 1)
w2.data.uniform_(-1, 1)


# NN
h=torch.matmul(x,w1)
y=torch.matmul(h,w2)
# loss=torch.sum(torch.abs(y-o))
loss=torch.sum((y-o)**2)
loss.backward()
w1.grad
w2.grad
torch.sum(h*(y-o),dim=0)*2


x_in=x.mul(w1)
h_t = h
y=h.mul(w2)

loss=torch.sum(torch.abs(y-torch.from_numpy(yN)))

loss.backward()

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
Q.backward()