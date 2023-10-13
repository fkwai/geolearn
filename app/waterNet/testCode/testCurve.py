import numpy as np
import matplotlib.pyplot as plt


d = np.linspace(0, 100, 100)
gL = 100
gK = 10
k=np.exp((d-gL)/gK)


fig, ax = plt.subplots(1, 1)
# ax.plot(k,d-gL)
ax.plot(d,k)
fig.show()

import torch
torch.tanh(torch.tensor([0.1,1.0, 2.0, 3.0]))
torch.cosh(torch.tensor([0.1,1.0, 2.0, 3.0,10]))

torch.log(torch.cosh(torch.tensor([19])))-torch.log(torch.cosh(torch.tensor([11])))