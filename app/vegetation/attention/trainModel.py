import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainBasin
import math
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm, dataTs2Range
import torch.optim as optim
import torchmetrics

from hydroDL import kPath

saveFolder = os.path.join(kPath.dirVeg, 'model', 'attention')

# load data
dataFile = os.path.join(saveFolder, 'data.npz')
data = np.load(dataFile)
x = data['x']
xc = data['xc']
y = data['y']
yc = data['yc']
tInd = data['tInd']
siteInd = data['siteInd']
subsetFile = os.path.join(saveFolder, 'subset.json')
with open(subsetFile, 'r') as fp:
    dictSubset = json.load(fp)

rho = 45
df = dbVeg.DataFrameVeg('singleDaily')


bS = 8
bL = 6
bM = 10
