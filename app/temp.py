# %% initial
from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
import torch
tRange = [20160401, 20180401]
testName = 'ecoRegion07_v2f1'
outName = 'ecoRegion0701_v2f1_Forcing'
out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion', outName)
df, yp, yt, sigma =
 master.test(out,tRange=tRange,subset=testName,doMC=100,reTest=True)
