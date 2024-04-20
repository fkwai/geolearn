import torch
import matplotlib.pyplot as plt

D = 10 # effective depth

x = torch.arange(0, 1, 0.01)
# y=torch.lgamma(x)
a=0.5
y=a*x**(a-1)

fig,ax=plt.subplots(1,1)
ax.plot(x,y)
fig.show()