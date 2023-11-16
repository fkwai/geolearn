from random import choices
import matplotlib.pyplot as plt
import numpy as np

c = [1, 0]
p = [0.8, 0.2]


kLst = np.arange(10, 500, 10)
fig, ax = plt.subplots(1, 1)
for k in kLst:
    for kk in range(int(1000/k)**2):
        ss = choices(c, p, k=k)
        _ = ax.plot(k, np.mean(ss), 'k*')
fig.show()

fig, ax = plt.subplots(1, 1)
ss = choices(c, p, k=2000)
kLst = [5, 10, 20, 50, 100, 200, 500]
for k in kLst:
    temp=np.array(ss).reshape(k,-1)    
    m=np.mean(temp,axis=0)
    _ = ax.plot(np.ones(m.shape)*k, m, 'k*')
fig.show()


file='/home/kuai/work/waterQuality/modelFull/rmTK-B200-QFT2C-rmYr5b0/testP-rmYr5b0-Ep40.npz'
npz=np.load(file)
yP=npz['yP']

variables=globals().keys()

import dill
dill.dump_session('temp.db')
dill.detect.trace(False)

for key, items in globals().items():
    print(key)
    dill.detect.errors(items)

dill.detect.errors(npz)

dill.detect.baditems(npz)
