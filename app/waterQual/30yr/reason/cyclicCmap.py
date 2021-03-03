import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sd = np.datetime64('2000-01-01')
ed = np.datetime64('2000-12-31')
t = pd.date_range(sd, ed)
td = t.dayofyear.values-1
fig, ax = plt.subplots(1, 1)
nt = td.max()
# tLst = ['2000-01-01', '2000-03-01', '2000-06-01', '2000-09-01']

tLst = ['2000-{:02d}-01'.format(m+1) for m in range(12)]
for k in range(len(tLst)):
    tt = pd.to_datetime(tLst[k]).dayofyear-1
    xx = np.cos(tt/nt*np.pi*2)
    yy = np.sin(tt/nt*np.pi*2)
    ax.plot([0, xx], [0, yy], 'k-')
    ax.text(xx, yy, tLst[k][5:])
x = np.cos(td/nt*np.pi*2)
y = np.sin(td/nt*np.pi*2)
ax.scatter(x, y, c=td, cmap='hsv',s=100)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_xticks([])
ax.set_aspect('equal', 'box')
fig.show()
