import numpy as np
import matplotlib.pyplot as plt

p = 10
s0 = 100
k = 0.2
n = 100
tLst = np.linspace(0, 1, n)
pp = p/n
kk = k/n
sLst = list()
dLst = list()

s = s0
for t in tLst:
    s = s+pp-kk*s
    sLst.append(s)
    dLst.append(p-k*s)
s1 = p/k*(1-np.exp(-tLst*k))+s0*np.exp(-tLst*k)

fig, ax = plt.subplots(1, 1)
ax.plot(sLst)
ax.plot(s1)
fig.show()
