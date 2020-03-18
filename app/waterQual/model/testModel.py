
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
import torch
import os
import json
import numpy as np

modelName = 'basinRef_first80'
modelFolder = os.path.join(kPath.dirWQ, 'model', modelName)
masterFile = os.path.join(modelFolder, 'master.json')
with open(masterFile, 'r') as fp:
    master = json.load(fp)

# load data
wqData = waterQuality.DataModelWQ(master['dataName'])
dataLst, statLst = wqData.transIn(subset=master['trainName'])
(x, xc, y, yc) = dataLst
(statX, statXC, statY, statYC) = statLst

# load model
nEpoch = 100
modelFile = os.path.join(modelFolder, 'model_ep{}'.format(nEpoch))
model = torch.load(modelFile)

# test model - point by point
yOut, ycOut = trainTS.testModel(model, x, xc, nyc)
