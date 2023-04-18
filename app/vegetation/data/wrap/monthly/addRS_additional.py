import numpy as np
from hydroDL import kPath
import os
import pandas as pd

# append forcings
outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
# load data
data = np.load(outFile)
x = data['x']
y = data['y']
xc = data['xc']
t = data['t']
varX = data['varX'].tolist()
varY = data['varY']
varXC = data['varXC'].tolist()
varX[varX.index('VV')] = 'vv'
varX[varX.index('VH')] = 'vh'

nir = x[:, :, varX.index('nir')]
red = x[:, :, varX.index('red')]
swir = x[:, :, varX.index('swir')]
vh = x[:, :, varX.index('vh')]
vv = x[:, :, varX.index('vv')]
ndvi = (nir - red) / (nir + red)
ndwi = (nir - swir) / (nir + swir)
nirv = nir * ndvi
vh_vv = vh - vv
xNew = np.dstack([x, ndvi, ndwi, nirv, vh_vv])
varX = varX + ['ndvi', 'ndwi', 'nirv', 'vh_vv']

outFile = os.path.join(kPath.dirVeg, 'model', 'data', 'trainData.npz')
np.savez(outFile, varY=varY, y=y, varX=varX, x=xNew, t=t, varXC=varXC, xc=xc)

