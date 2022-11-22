
import json
import numpy as np
from numpy import double
import pandas as pd
from hydroDL.data import camels,
import importlib
import matplotlib.pyplot as plt
from hydroDL.post import axplot

importlib.reload(camels)
dfInfo = camels.dfInfo
dfG = camels.readAttr(varLst=camels.varG)

plt.subplots(4, 4)

sn = 1e-8
for var in camels.varG:
    fig, axes = plt.subplots(2, 1)
    axes[0].hist(dfG[var].values)
    axes[1].hist(np.log(dfG[var].values+sn))
    axes[0].set_title(var)
    fig.show()
camels.varG[-1]

a=dfG['geol_permeability'].values
plt.hist(a)
plt.show()