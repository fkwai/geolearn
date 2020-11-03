from hydroDL.post import axplot, figplot
from hydroDL.new import fun
from hydroDL.app import waterQuality
import importlib
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
from scipy.special import gamma

# gamma
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# kLst = [0.5, 1, 2]
kLst = [50]
theLst = [0.1, 0.5, 1, 3]
sLst = [20]
for k in kLst:
    for the in theLst:
        for s in sLst:
            x = np.linspace(0, s, 365)
            b = 1/the
            a = k
            q = b**a*x**(a-1)*np.exp(-b*x)/gamma(a)
            t = np.arange(365)
            ax.plot(t, q, label='k={} the={} s={}'.format(k, the, s))
            np.mean(q)*s
ax.legend()
fig.show()

# beta
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
aLst = [0.5, 1, 5, 10]
bLst = [0.5, 1, 5, 10]
for a in aLst:
    for b in bLst:
        x = np.linspace(0, 1, 365)
        q = gamma(a+b)/gamma(a)/gamma(b)*x**(a-1)*(1-x)**(b-1)
        ax.plot(t, q, label='a={} b={}'.format(a, b))
ax.legend()
fig.show()

