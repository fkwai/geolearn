from astropy.timeseries import LombScargle
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time


# test
x = np.arange(5)
y = np.random.random(5)
ls = LombScargle(x, y)
freq = 2/5
p = ls.model_parameters(freq)
mat = ls.design_matrix(freq)
yp = ls.model(x, freq)
power = ls.power(freq, normalization='psd')
offset = ls.offset()

a = np.fft.fft(y)

yp1 = p[0]+p[1]*np.sin(2*np.pi*freq*x)+p[2]*np.cos(2*np.pi*freq*x)+ls.offset()

mat
np.sin(2*np.pi*freq*x)
np.cos(2*np.pi*freq*x)

yp1 = yp*0
for f in list(x/5)[1:]:
    f
    yp1 = yp1+ls.model(x, f)

xf = ls.power(freq, normalization='psd')*2 / \
    ls.power(freq, normalization='model')
np.sqrt(xf)
xref = ls.power(freq, normalization='psd')*2 / ls.power(freq)
np.sqrt(xref)

xfM = np.sum((yp1-y)**2)
np.sum((np.mean(y)-y)**2)
