from scipy.stats import norm, invgamma
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots(1,1)

x = np.linspace(-5, 5)
a = 1
b = 0.5
y1 = invgamma.pdf(x, a, loc=0, scale=b)
y2 = norm.pdf(x,loc=0,scale=1)


ax.plot(x,y1,'-r')
ax.plot(x,y2^2,'-b')
fig.show()