import numpy as np


def calErr(pred, obs, rmExt=True):
    a = obs
    b = pred
    if len(obs) == 0:
        return (np.nan, np.nan)
    if rmExt is True and len(a) != 0:
        aV = a[a < np.nanpercentile(a, 95)]
        aV = aV[aV > np.nanpercentile(a, 5)]
        ul = np.mean(aV)+np.std(aV)*5
        a[a > ul] = np.nan
    indV = np.where(~np.isnan(a))
    if len(indV) > 0:
        rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
        corr = np.corrcoef(a[indV], b[indV])[0, 1]
        return (rmse, corr)
    else:
        return (np.nan, np.nan)
