from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataName = 'nbW'
wqData = waterQuality.DataModelWQ(dataName)
nd = pd.DatetimeIndex(wqData.info['date']).dayofyear.values
rho = wqData.rho
ns = len(nd)
if wqData.freq == 'D':
    dd = np.flipud(np.arange(0, rho))
if wqData.freq == 'W':
    dd = np.flipud(np.arange(0, rho))*7
tMat = np.tile(np.expand_dims(nd, axis=1), [1, rho])
tMat = (tMat-dd)/365
sinT = np.sin(2*np.pi*tMat).swapaxes(0, 1).astype(np.float32)
cosT = np.cos(2*np.pi*tMat).swapaxes(0, 1).astype(np.float32)
matT = np.stack([sinT, cosT], axis=2)
wqData.f = np.concatenate([wqData.f, matT], axis=2)
wqData.varF.append('sinT')
wqData.varF.append('cosT')

