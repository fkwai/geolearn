import torch
import numpy as np
import scipy.special as sp
from torch.nn import Parameter
import time


def func(state, input, param):
    return state + input * param


nt = 10
k = torch.tensor(1.0, requires_grad=True)
I = torch.tensor(0.0, requires_grad=True)
H = []
Q = []
p = 5
rhoW = 2
rhoB = 3

s0 = I
with torch.no_grad():
    for t in range(rhoW):
        s0 = func(s0, t, k)
for t in range(rhoW, rhoW + rhoB):
    s1 = func(s0, t, k)
    s0 = s1
    H.append(s1.detach())
    Q.append(s1)
for t in range(rhoW + rhoB, nt):
    s0 = H[0]
    for i in range(rhoB):
        s1 = func(s0, i + t - rhoB + 1, k)
        s0 = s1
    _ = H.append(s1.detach())
    _ = H.pop(0)
    _ = Q.append(s1)
    s1.backward(retain_graph=True)
    k.grad
    _ = k.grad.zero_()

# merged
s = I
with torch.no_grad():
    for t in range(rhoW):
        s = func(s, t, k)
H = [s]
for t in range(rhoW, nt):
    if t < rhoW + rhoB:
        s = func(s, t, k)
    else:
        s = H[0]
        for i in range(rhoB):
            s = func(s, i + t - rhoB + 1, k)
        _ = H.pop(0)
    _ = H.append(s.detach())
    _ = Q.append(s)
    if t >= rhoW + rhoB:
        
    s.backward(retain_graph=True)
    k.grad
    _ = k.grad.zero_()


def print_graph(g, level=0):
    if g == None:
        return
    print('*' * level * 4, g)
    for subg in g.next_functions:
        print_graph(subg[0], level + 1)


print_graph(Q[4].grad_fn, 0)
