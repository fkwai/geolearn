
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
import pandas as pd
import json
import os

d1 = waterQuality.DataModelWQ('sbWT')
d2 = waterQuality.DataModelWQ('sbWTQ')

d1.q.shape
d2.q.shape

np.sum(d1.c-d2.c)

len(d2.varC)
d2.c.shape

len(d2.varF)
d2.f.shape

np.nansum(d1.f[:,:,:-3]-d2.f)