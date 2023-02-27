from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd

dataName = 'dbAll'
codeLst = usgs.varC
DF = dbBasin.DataFrameBasin(dataName)

code = '00915'
indC=DF.varC.index(code)