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
import pandas as pd

code = '00915'
label = 'QFT2C'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
dataName = '{}-{}'.format(code, 'B200')
DF = dbBasin.DataFrameBasin(dataName)
matObs = DF.extractT([code])
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)


outName='00915-B200-QFT2C-rmYr5b0-d25-h64-rho365-nl1'
yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=500)

def testModel(code, dr, hs, rho, nLayer, ep=500):
    dataName = '{}-{}'.format(code, 'B200')
    outName = '{}-{}-{}-d{:.0f}-h{}-rho{}-nl{}'.format(
        dataName, label, trainSet, dr * 100, hs, rho, nLayer
    )
    yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=ep)
    corr = utils.stat.calCorr(yP2, obs2)
    return corr


drLst = [0.25, 0.5, 0.75]
hsLst = [64, 256, 512]
rhoLst = [365, 1000, 2000]
nLayerLst = [1, 2]
epLst = [100, 300, 500]
# codeLst = ['00618','00915','00955']

columns = ['dr', 'hs', 'rho', 'nLayer', 'ep']
tab = pd.DataFrame(columns=columns + ['corr'])

for dr in drLst:
    for hs in hsLst:
        for rho in rhoLst:
            for nLayer in nLayerLst:
                for ep in [100, 300, 500]:
                    try:
                        corr = testModel(code, dr, hs, rho, nLayer,ep=ep)
                        temp = {
                            'dr': dr,
                            'hs': hs,
                            'rho': rho,
                            'nLayer': nLayer,
                            'ep': ep,
                            'corr': np.nanmedian(corr),
                        }
                        tab = tab.append(temp, ignore_index = True)

                    except:
                        print(code, dr, hs, rho, nLayer)
tab.to_csv('temp.csv')