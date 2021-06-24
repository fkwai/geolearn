import pandas as pd
from hydroDL.data import dbBasin
import numpy as np
import matplotlib.pyplot as plt
import os
from hydroDL.master import basinFull
import torch
from hydroDL.model import rnn, crit, trainBasin

outLst = ['G200Norm', 'G400Norm']
epLst = list(range(100, 1100, 100))

# save
for outName in outLst:
    for ep in epLst:
        outFolder = basinFull.nameFolder(outName)
        modelFile = os.path.join(outFolder, 'model_ep{}'.format(ep))
        model = torch.load(modelFile)
        modelStateFile = os.path.join(outFolder, 'modelState_ep{}'.format(ep))
        torch.save(model.state_dict(), modelStateFile)
