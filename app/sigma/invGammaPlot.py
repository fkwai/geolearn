
from scipy.special import gamma, factorial
import matplotlib.pyplot as plt
import numpy as np

def invGamma(x, a, b):
    f = np.power(b, a)/gamma(a)*np.power(1/x, a+1)*np.exp(-b/x)
    return f


fig, axes = plt.subplots(5, 1)
x = np.linspace(0, 1, 1000)
for i, a in enumerate([2, 3, 4, 5, 6]):
    for j, b in enumerate([0.1, 0.25, 0.5, 0.75, 1]):
        print(a, b)
        f = invGamma(x, a, b)
        axes[j].plot(x, f, label='a = {}'.format(a))
        axes[j].legend()
        axes[j].set_title('b = {}'.format(b))
fig.show()
