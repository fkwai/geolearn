from scipy.stats import invgamma
import matplotlib.pyplot as plt

import numpy as np

x = np.linspace(invgamma.ppf(0.01, a),
                invgamma.ppf(0.99, a), 100)
rv = invgamma(a)
fig, ax = plt.subplots(1, 1)

ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
