import os
import time
import json
import numpy as np
import pandas as pd
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.post import plot
import matplotlib.pyplot as plt
from datetime import datetime as dt


caseName = 'temp'
nEpoch = 200
modelFolder = os.path.join(kPath.dirWQ, 'modelA', caseName)
saveFile = os.path.join(modelFolder, 'statResult_Ep{}.npz'.format(nEpoch))
npf = np.load(saveFile)
nc = 21

matRho1 = matRho1, matRho2 = matRho2, matRmse1 = matRmse1, matRmse2 = matRmse2, matN1 = matN1, matN2 = matN2
dataBox = list()
for k in range(nc):
    temp = [npf['matRho1'][:, k], npf['matRho2'][:, k]]
    dataBox.append(temp)
fig=plot.plotBoxFig(dataBox,sharey=False)
fig.show()