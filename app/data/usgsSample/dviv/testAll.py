from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os
import argparse
import socket
import json
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot

dataName='dbAll'
DF = dbBasin.DataFrameBasin(dataName)

a=DF.q[:,:,0]
indQ=DF.varC.index('00060')
b=DF.c[:,:,indQ]

np.nansum(a-b)

matV=(~np.isnan(a)) & (~np.isnan(b))
# fig,ax=plt.subplots(1,1)
# ax.plot(a[matV],b[matV],'*')
# fig.show()

