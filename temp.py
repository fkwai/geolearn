import numpy as np
import matplotlib.pyplot as plt

nG = 5
nP = 100
nV = 3
the = 10
data = np.random.randint(0, 15, [nG*nP, nV*2+1])

matX = np.reshape(data[:, :nV], [nG, nP, nV])
matC = np.reshape(data[:, nV:nV*2], [nG, nP, nV])
matY = np.reshape(data[:, -1], [nG, nP, 1])
matThe = matC < the

fig, axes = plt.subplots(nG, nV)
for k in range(nG):
    for j in range(nV):
        ind = matThe[k, :, j]
        x = matX[k, ind, j]
        y = matY[k, ind]
        axes[k, j].plot(x, y, '*')
        axes[k, j].set_title('group {} variable {}'.format(k, j))
plt.tight_layout()
fig.show()
