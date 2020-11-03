from scipy.stats import gamma
import numpy as np


def fdc(t, k, the=10):
    return gamma.pdf(t, k, scale=the)


def poisson(k, a):
    p = np.exp(a)*a**k/np.math.factorial(k)
    return p


def kate(t, r, c0=1, cw=10):
    p = np.exp(-t/r/365)
    ct = c0*p+cw*(1-p)
    return ct
