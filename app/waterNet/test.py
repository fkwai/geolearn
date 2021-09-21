import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin

# test case
siteNo = '07144100'
df = dbBasin.readSiteTS(siteNo, ['pr', 'runoff', 'pet'])
P = df['pr'].values
Q = df['runoff'].values
E = df['pet'].values

n = 100
nt = len(Q)
i = np.random.random_sample(size=n)
o = np.random.random_sample(size=n)
s = np.zeros(n)
sMat = np.zeros([nt, n])
out = np.zeros(nt)
for k in range(nt):
    p = P[k]
    s = p*i-s*o
    s[s < 0] = 0
    # s[s > 6] = 6
    sMat[k, :] = s
    out[k] = np.mean(s*o)

fig, axes = plt.subplots(3, 1)
axes[0].plot(P)
axes[1].plot(out, '-r')
axes[1].plot(Q)
fig.show()
