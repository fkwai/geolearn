import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np


# importlib.reload(dbBasin.io)
importlib.reload(dbBasin.dataModel)
importlib.reload(dbBasin)
importlib.reload(basinFull)


dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)
outName = '{}-B10'.format(dataName)

yP, ycP = basinFull.testModel(outName, batchSize=20, reTest=True)
yO = dm.q[:, :, 1]

k = 100

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, dm.t, [yO[:, k], yP[:, k]])
fig.show()
