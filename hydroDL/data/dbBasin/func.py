from hydroDL import kPath, utils
import os
import time
import pandas as pd
import numpy as np
import json
"""
additional functions dataModel processing
"""

def pickByYear(t, yrIn, pick=True):
    yr = t.astype('M8[Y]').astype(str).astype(int)
    bp = np.in1d(yr, yrIn)
    if pick:
        indT = np.where(bp)[0]
    else:
        indT = np.where(~bp)[0]
    tOut = t[indT]
    return tOut
