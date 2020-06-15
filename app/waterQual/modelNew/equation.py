import numpy as np
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


c0 = 0
c1 = 1
r = 0.5


def func(t):
    rr = 10**r
    c = c0 * np.exp(-rr*t)*rr + c1*(1-np.exp(-rr*t))
    return(c)


t = np.arange(512)/512
fig, ax = plt.subplots(1, 1)
for r in np.linspace(-1, 1, 11):
    ax.plot(t, func(t))
fig.show()
