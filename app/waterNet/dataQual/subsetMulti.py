from hydroDL.master import slurm
from hydroDL.master import basinFull
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

# count for code
code = '00600'
codeLst = ['00600', '00618', '00915', '00945', '00955']
pLst = [100, 75, 50, 25]
nyLst = [6, 8, 10]

for code in codeLst:
    for ny in nyLst:
        for p in pLst:
            label = 'QFPRT2C'
            trainSet = '{}-n{}-p{}-B10'.format(code, ny, p)
            varX = dbBasin.label2var(label.split('2')[0])
            mtdX = dbBasin.io.extractVarMtd(varX)
            varY = [code]
            mtdY = dbBasin.io.extractVarMtd([code])
            varXC = gageII.varLst
            mtdXC = dbBasin.io.extractVarMtd(varXC)
            varYC = None
            mtdYC = [code]
            outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
            dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                         nEpoch=500, batchSize=[365, 20], nIterEp=50,
                                         varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                         mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
            cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
            slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
            # basinFull.trainModel(outName)
