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


def createMaskByT(sd, ed, ns, tSub):
    # create a mask where TRUE WILL BE NAN later
    sd = np.datetime64(sd)
    ed = np.datetime64(ed)
    t = pd.date_range(sd, ed).values.astype('datetime64[D]')
    ind, indT = utils.time.intersect(tSub, t)
    mask = np.ones([len(t), ns]).astype(bool)
    mask[indT, :] = False
    return mask
