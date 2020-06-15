
import numpy as np
import matplotlib.pyplot as plt
import scipy

n = 365*5
t = np.linspace(0, n, n)
x = 3 * np.sin(2*t*2*np.pi/365)
# x = np.array([1, 1, 1, 1])
n = len(x)
# w = np.exp(complex(0, -2*np.pi/n))
# # f = 2*np.pi/np.linspace(1/n, 1, n)
# M = np.ndarray([n, n], dtype=complex)
# for j in range(n):
#     for i in range(n):
#         M[j, i] = w**(j*i)
# F = M/np.sqrt(n)
# Ft = 1/M/np.sqrt(n)
# y = F.dot(x)
freq = np.fft.fftfreq(x.size, np.mean(np.ediff1d(t)))

fig, axes = plt.subplots(3, 1)
axes[0].plot(t, x, '--*')
axes[1].plot(freq, np.fft.fft(x)**2, '--*')
axes[2].plot(1/freq, np.fft.fft(x)**2, '--*')
fig.show()

n = 1000
t = np.linspace(0, 4*np.pi, n)
x = 3 * np.sin(2*t + 0.5*np.pi)
y = scipy.fft(x)
fig, axes = plt.subplots(2, 1)
axes[0].plot(t, x, '--*')
axes[1].plot(1/np.arange(n-1), abs(y[1:]), '--*')
fig.show()
