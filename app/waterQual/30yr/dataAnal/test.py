import numpy as np
import matplotlib.pyplot as plt
from hydroDL import utils

n = 10000

# x = np.random.random(n)
x1 = np.random.normal(loc=0, scale=1.0, size=5000)
x2 = np.random.normal(loc=0, scale=2.0, size=500)

x = np.concatenate([x1, x2])
fig, ax = plt.subplots(1, 1)
ax.hist(x, bins=50)
fig.show()
