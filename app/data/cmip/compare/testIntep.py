from scipy import interpolate
import hydroDL.utils.stat
from scipy import interpolate
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import hydroDL.utils.ts

y = np.arange(9)
x = np.arange(9)
yy = np.arange(1, 9, 3)
xx = np.arange(1, 9, 3)
xv, yv = np.meshgrid(x, y, indexing='xy')

z = np.random.rand(9,9)

interp = interpolate.RegularGridInterpolator(
    (y, x), z, bounds_error=False,method='slinear')
zz  = interp((yy, xx))

temp=z[:3,:3]