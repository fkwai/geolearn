import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import scipy

# First define some input parameters for the signal:

A = 2.
w = 10
# w = 2*np.pi/365
phi = 0.5 * np.pi
nin = 1000
nout = 1000
r = np.random.rand(nin)
x = np.linspace(0.01, 10, nin)
# x = x[r >= 0.9]
y = A * np.sin(w*x+phi)
f = np.linspace(0.01, 10, nout)
pgram = signal.lombscargle(x, y, f)

fig, axes = plt.subplots(2, 1)
axes[0].plot(x, y, '--*')
axes[1].plot(f, pgram)
fig.show()

# try to use fft
f2, p2 = signal.periodogram(x)
# sp = np.fft.fft(y)
fig, axes = plt.subplots(2, 1)
axes[0].plot(x, y, '--*')
axes[1].plot(f2, p2)
fig.show()

