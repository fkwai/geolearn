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
    indT = np.where(bp)[0] if pick else np.where(~bp)[0]
    tOut = t[indT]
    return tOut


def pickByDay(t, dBase, dSel, pick=True):
    if type(dSel) is not list:
        dSel = [dSel]
    sd = np.datetime64('1980-01-01')
    d = (t-sd).astype(int)
    r = d % dBase
    bp = np.in1d(r, dSel)
    indT = np.where(bp)[0] if pick else np.where(~bp)[0]
    tOut = t[indT]
    return tOut


def pickRandT(t, rate, seed=0, pick=True):
    rng = np.random.default_rng(seed)
    tp = rng.choice(t, int(len(t)*rate))
    if pick is True:
        return tp
    else:
        b = ~np.in1d(t, tp)
        return t[b]


def createMaskByT(sd, ed, ns, tSub):
    # create a mask where TRUE WILL BE MASK OUT later
    sd = np.datetime64(sd)
    ed = np.datetime64(ed)
    t = pd.date_range(sd, ed).values.astype('datetime64[D]')
    ind, indT = utils.time.intersect(tSub, t)
    mask = np.ones([len(t), ns]).astype(bool)
    mask[indT, :] = False
    return mask


def localNorm(DF, codeLst=None, subset=None):
    if codeLst is None:
        codeLst = DF.varC.copy()
    if subset is None:
        subset = 'all'
    dataTrain = DF.extractSubset(DF.c, subsetName=subset)
    for code in codeLst:
        ic = DF.varC.index(code)
        temp = dataTrain[:, :, ic].copy()
        matB = ~np.isnan(temp)
        count = np.sum(matB, axis=0)
        indS = np.where(count < 50)[0]
        temp[:, indS] = np.nan
        mean = np.nanmean(temp, axis=0)
        std = np.nanstd(temp, axis=0)
        out = (DF.c[:, :, ic]-mean)/std
        DF.c = np.concatenate([DF.c, out[:, :, None]], axis=2)
        DF.varC.append(code+'-N')
        DF.g = np.concatenate([DF.g, mean[:, None]], axis=1)
        DF.varG.append(code+'-M')
        DF.g = np.concatenate([DF.g, std[:, None]], axis=1)
        DF.varG.append(code+'-S')
    return DF
