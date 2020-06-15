import numpy as np
import matplotlib.pyplot as plt
from numeric import ode, es

# example from text book
y0 = 1
y1 = 4
a = 0
b = 1
n = 10
h = (b-a)/(n+1)


def F(w):
    ww = w.copy()
    ww = np.insert(ww, 0, y0)
    ww = np.append(ww, y1)
    y = np.zeros(n)
    for k in range(n):
        w0 = ww[k]
        w1 = ww[k+1]
        w2 = ww[k+2]
        y[k] = w0+w2-(2+h**2)*w1+h**2*w1**2
    return y


def DF(w):
    diag = 2*h**2*w - (2+h**2)
    mat = np.eye(n)*diag
    rng = np.arange(n-1)
    mat[rng, rng+1] = 1
    mat[rng+1, rng] = 1
    return mat


x = np.zeros(n)
for k in range(10):
    s = es.gaussElimination(DF(x), -F(x))
    x = x+s
    # print(x)

fig, ax = plt.subplots(1, 1)
ax.plot(np.arange(n), x, 'b')
fig.show()


# test in ivp


# def dF(t, y):
#     return np.array([y[1]-y[1]**2, y[0]])


# def throw(y0, t1, dt=0.1):
#     ny = 2
#     t = np.arange(0, t1, 0.1)
#     y = np.ndarray([len(t), ny])
#     y[0, :] = y0
#     for k in range(1, len(t)):
#         yy = y[k-1, :]
#         tt = t[k-1]
#         dt = t[k]-t[k-1]
#         y[k, :] = ode.forward1(tt, yy, dt, dF)
#     return t, y


# v0 = (y[0]-y0)/h
# t, y2 = throw([3, 1], 1)
# fig, ax = plt.subplots(1, 1)
# ax.plot(t, y, 'b')
# ax.plot(t, y2[:, 1], 'r')
# fig.show()
