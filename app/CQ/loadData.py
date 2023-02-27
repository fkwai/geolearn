from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd

dataName = '00955-B200'
DF = dbBasin.DataFrameBasin(dataName)

# tsmap of C-Q and C-T
