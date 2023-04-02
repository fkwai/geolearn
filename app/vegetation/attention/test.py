import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os

dataName='singleDaily'
importlib.reload(hydroDL.data.dbVeg)
df=dbVeg.DataFrameVeg(dataName)

dm=dbVeg.DataModelVeg(df, subsetName='all')

