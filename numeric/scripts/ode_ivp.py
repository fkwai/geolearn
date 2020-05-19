import numpy as np
import matplotlib.pyplot as plt

from numeric import ode

# ODE system using euler forwarding


def F(t):
    return np.array([t*np.exp(-2*t), np.exp(-t)])


def dF(t, y):
    return np.array([-y[0]*2+y[1]**2, y[0]-y[1]-t*y[1]**2])


ny = 2
t = np.arange(0, 10, 0.01)
y = np.ndarray([len(t), ny])
y[0, :] = [0, 1]
for k in range(1, len(t)):
    yy = y[k-1, :]
    tt = t[k-1]
    dt = t[k]-t[k-1]
    y[k, :] = ode.forward2(tt, yy, dt, dF)

fig, ax = plt.subplots(1, 1)
f = F(t)
ax.plot(t, f[0, :], 'b')
ax.plot(t, f[1, :], 'b')
ax.plot(t, y[:, 0], 'r')
ax.plot(t, y[:, 1], 'r')
fig.show()

# throw

g = 10
y0 = [10, 0]


def F(t):
    return np.array([-g*t+y0[0], -1/2*g*t**2+y0[0]*t+y0[1]]).T


def dF(t, y):
    return np.array([-g, y[0]])


def throw(y0, t1, dt=0.01):
    ny = 2
    t = np.arange(0, t1, 0.01)
    y = np.ndarray([len(t), ny])
    y[0, :] = y0
    for k in range(1, len(t)):
        yy = y[k-1, :]
        tt = t[k-1]
        dt = t[k]-t[k-1]
        y[k, :] = ode.forward1(tt, yy, dt, dF)
    return t, y


t, y = throw(y0, 1)
fig, ax = plt.subplots(1, 1)
f = F(t)
ax.plot(t, f, 'b')
ax.plot(t, y, 'r')
fig.show()
