
from hydroDL.data import dbBasin
import json
import os
from hydroDL import kPath
import numpy as np
import pandas as pd

sd = '1982-01-01'
ed = '2018-12-31'
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictY28N5.json')) as f:
    dictSite = json.load(f)

len(dictSite['comb'])
len(dictSite['rmTK'])

DF = dbBasin.DataFrameBasin.new(
    'Y28N5', dictSite['comb'], sdStr=sd, edStr=ed)

DF = dbBasin.DataFrameBasin.new(
    'Y28N5rmTK', dictSite['rmTK'], sdStr=sd, edStr=ed)

# before after 2010
DF = dbBasin.DataFrameBasin('Y28N5')
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')

DF = dbBasin.DataFrameBasin('Y28N5rmTK')
DF.saveSubset('B10', ed='2009-12-31')
DF.saveSubset('A10', sd='2010-01-01')