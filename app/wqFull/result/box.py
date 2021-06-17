
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull

dataName = 'G400Norm'
outName = dataName
trainSet = 'rmRT20'
testSet = 'pkRT20'

DF = dbBasin.DataFrameBasin(outName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet,ep=200, reTest=True)
