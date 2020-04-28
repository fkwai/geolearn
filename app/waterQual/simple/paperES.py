import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataFile = os.path.join(kPath.dirWQ, 'tempData', 'es403723r_si_002.csv')
df = pd.read_csv(dataFile, encoding='ISO-8859-1')
t = pd.to_datetime(df['date'])
df.columns
field = 'Si_(29)  ppb'
fig, ax = plt.subplots(1, 1,figsize=(16,4))
axplot.plotTS(ax, t, df[field], styLst=['-*'])
ax.set_title(field)
fig.show()
