import pandas as pd
import numpy as np


def data2Monthly(data, t, func='nanmean'):
    # data in [...,nt]
    pdT = pd.to_datetime(t)
    ymAry = (pdT.year*100+pdT.month).values
    ymUni = np.unique(ymAry)
    tempT = pd.to_datetime(ymUni.astype(str), format='%Y%m')
    outT = tempT.values.astype('datetime64[M]')
    out = np.ndarray(data.shape[:-1]+(len(ymUni),))
    for k, ym in enumerate(ymUni):
        ind = np.where(ymAry == ym)[0]
        tempG = data[..., ind]
        f = getattr(np, func)
        out[..., k] = f(tempG, axis=-1)
    return out, outT


def data2Climate(data, t, func='nanmean'):
    # data in [...,nt]
    pdT = pd.to_datetime(t)
    ndAry = pdT.dayofyear.values
    ndUni = np.array(range(365))+1
    out = np.ndarray(data.shape[:-1]+(len(ndUni),))
    for k, nd in enumerate(ndUni):
        ind = np.where(ndAry == nd)[0]
        tempG = data[..., ind]
        f = getattr(np, func)
        out[..., k] = f(tempG, axis=-1)
    return out, ndUni
