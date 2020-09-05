from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib

import warnings
warnings.simplefilter('always')

dataName = 'sbW'
varX = gridMET.varLst+ntn.varLst
varY = ['00060']
varYC = ['00410']
subset = '00410-Y1'
saveName = 'temp'
caseName = basins.wrapMaster(
    dataName=dataName, trainName=subset, batchSize=[None, 100],
    outName=saveName, varX=varX, varY=varY, varYC=varYC)
basins.trainModelTS(caseName)
