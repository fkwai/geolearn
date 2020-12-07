import numpy as np


def calErr(pred, obs, rmExt=False):
    a = obs
    b = pred
    if len(obs) == 0:
        return (np.nan, np.nan)
    if rmExt is True and len(a) != 0:
        aV = a[a < np.nanpercentile(a, 95)]
        aV = aV[aV > np.nanpercentile(a, 5)]
        ul = np.mean(aV)+np.std(aV)*5
        a[a > ul] = np.nan
    indV = np.where(~np.isnan(a) & ~np.isnan(b))
    if len(indV) > 0:
        rmse = np.sqrt(np.nanmean((a[indV]-b[indV])**2))
        corr = np.corrcoef(a[indV], b[indV])[0, 1]
        return (rmse, corr)
    else:
        return (np.nan, np.nan)


def calNash(pred, obs):
    # data in [nT,nS]
    nash = 1-np.nansum((pred-obs)**2, axis=0) / \
        np.nansum((obs-np.nanmean(obs, axis=0))**2, axis=0)
    return nash


def calRmse(pred, obs):
    # data in [nT,nS]
    rmse = np.sqrt(np.nanmean((pred-obs)**2, axis=0))
    return rmse


def calCorr(pred, obs):
    # data in [nT,nS]
    [nT, nS] = pred.shape
    corr = np.full([nS], np.nan)
    for k in range(nS):
        a = pred[:, k]
        b = obs[:, k]
        indV = np.where(~np.isnan(a) & ~np.isnan(b))
        corr[k] = np.corrcoef(a[indV], b[indV])[0, 1]
    return corr
