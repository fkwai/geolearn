import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.master import basinFull
from hydroDL.master import slurm

code = '00915'
dataName = '{}-{}'.format(code, 'B200')
DF = dbBasin.DataFrameBasin(dataName)


def testModel(code, dr, hs, rho, nLayer,ep=500):
        outName = '{}-{}-{}-d{:.0f}-h{}-rho{}-nl{}'.format(
            dataName, label, trainSet, dr * 100, hs, rho, nLayer
        )
        yP1, ycP1 = basinFull.testModel(outName, DF=DF,testSet=trainSet, ep=ep)
        yP2, ycP2 = basinFull.testModel(outName, DF=DF,testSet=testSet, ep=ep)
        return outName

drLst = [0.25, 0.5, 0.75]
hsLst = [64, 256, 512]
rhoLst = [365, 1000, 2000]
nLayerLst = [1, 2]

# codeLst = ['00618','00915','00955']

# for label in labelLst:
label = 'QFT2C'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
for dr in drLst:
    for hs in hsLst:
        for rho in rhoLst:
            for nLayer in nLayerLst:
                for ep in [100,300, 500]:
                    try:
                        outName=testModel(code, dr, hs, rho, nLayer,ep=ep)
                    except:
                        print(code, dr, hs, rho, nLayer)

print('finished')
