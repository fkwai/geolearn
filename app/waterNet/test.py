import numpy as np
import matplotlib.pyplot as plt

p = 1
n = 10
i = np.random.lognormal(size=n)
w = np.random.lognormal(size=(n, n))
o = np.random.lognormal(size=n)
s = np.ones(n)*100

# update w
row, col = np.diag_indices_from(w)
# w[row, col] = 0
# w[row, col] = 1-np.sum(w, axis=1)

outLst = list()

for k in range(10):
    s = np.matmul(s, w)+s-s*o 
    # s[s < 0] = 0
    outLst.append(np.sum(s*o))

s = np.ones(n)
s = np.matmul(s, w)
np.sum(s)

a,b=np.linalg.eig(w)

fig, ax = plt.subplots(1, 1)
# ax.set_yscale('log')
ax.plot(outLst)
fig.show()

# fig, ax = plt.subplots(1, 1)
# ax.imshow(w)
# fig.show()
