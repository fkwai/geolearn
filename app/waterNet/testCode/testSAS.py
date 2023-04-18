import torch
import numpy as np
import scipy.special as sp
from torch.nn import Parameter
import time

nt = 10
nh = 2
x = torch.tensor([2, 4, 5, 6, 1])
x = torch.rand(nt)

a = torch.tensor(2)
b = torch.tensor(5)

torch.exp(
    torch.lgamma(a + b)
    - torch.lgamma(b)
    - torch.lgamma(a)
    + (a - 1) * torch.log(x)
    + (b - 1) * torch.log(1 - x)
)

nt = 10
k = Parameter(torch.tensor(0.9))
H = []
Q = []
p = 5
rho = 3
for t in range(nt):
    if t == 0:
        s0 = Parameter(torch.tensor(10.0))
    else:
        s0 = H[-1]
    s1 = s0 + t * k
    if len(H) >= rho:
        _ = H[t - rho].detach_()
    H.append(s1)
s1.backward()
k.grad


def func(state, input, param):
    return state + input * param


nt = 10
k = torch.tensor(1.0, requires_grad=True)
I = torch.tensor(0.0, requires_grad=True)
H = []
Q = []
p = 5
rho = 3

s0 = I
for t in range(rho):
    s1 = func(s0, t, k)
    s0 = s1
    H.append(s1.detach())
    Q.append(s1)
for t in range(rho, nt):
    s0 = H[0]
    for i in range(rho):
        s1 = func(s0, i + t - rho + 1, k)
        s0 = s1
    _ = H.append(s1.detach())
    _ = H.pop(0)
    _ = Q.append(s1)
    s1.backward(retain_graph=True)
    k.grad
    _ = k.grad.zero_()


def print_graph(g, level=0):
    if g == None:
        return
    print('*' * level * 4, g)
    for subg in g.next_functions:
        print_graph(subg[0], level + 1)


print_graph(Q[4].grad_fn, 0)

nt = 10
k = torch.tensor(0.9, requires_grad=True)
H = []
Q = []
p = 5
rho = 3
k.grad.zero_()  # zero out the gradient once before the loop
for t in range(nt):
    if t == 0:
        s0 = torch.tensor(10.0, requires_grad=True)
    else:
        s0 = H[-1]
    s1 = s0 + t * k
    if len(H) >= rho:
        with torch.no_grad():
            _ = H[t - rho]
    H.append(s1)
    s1.backward()  # no need to use retain_graph=True
    print(k.grad)
