import numpy as np
import matplotlib.pyplot as plt

nt = 365
pos = np.array(range(nt))
n = 365*10
d = 512
i = np.array(range(d))
vMat = np.ndarray([nt, d])
for j, p in enumerate(pos):
    v = np.sin(p/(n**(i/d)))
    vMat[j, i] = v

# fig, ax = plt.subplots(1, 1)
# ax.plot(i, vMat[10:])
# fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(pos, vMat[:, 200])
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(i, n**(i/d))
fig.show()
