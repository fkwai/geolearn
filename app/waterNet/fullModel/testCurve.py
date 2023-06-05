import numpy as np
import matplotlib.pyplot as plt


d = np.linspace(0, 100, 100)
gL = 100
gK = 0.01
k=np.exp(gK * (d-gL))


fig, ax = plt.subplots(1, 1)
ax.plot(k,d)
fig.show()


