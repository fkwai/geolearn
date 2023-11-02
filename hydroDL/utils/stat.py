import numpy as np
import warnings


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


def calStat(pred, obs):
    if np.isnan(pred).all() or np.isnan(obs).all():
        return dict(Bias=np.nan, RMSE=np.nan, NSE=np.nan, Corr=np.nan)
    else:
        nash = calNash(pred, obs)
        rmse = calRmse(pred, obs)
        corr = calCorr(pred, obs)
        bias = calBias(pred, obs)
        outDict = dict(Bias=bias, RMSE=rmse, NSE=nash, Corr=corr)
        return outDict


# def calNash(pred, obs):
#     # data in [nT,nS]
#     nash = 1-np.nansum((pred-obs)**2, axis=0) / \
#         np.nansum((obs-np.nanmean(obs, axis=0))**2, axis=0)
#     return nash
def calNash(pred, obs):
    mask = np.isnan(obs) | np.isnan(pred)
    A = pred.copy()
    B = obs.copy()
    A[mask] = np.nan
    B[mask] = np.nan
    mA = A - B
    mB = B - np.nanmean(B, axis=0)
    p1 = np.nansum(mA**2, axis=0)
    p2 = np.nansum(mB**2, axis=0)
    return 1-p1/p2

def calRmse(pred, obs):
    # data in [nT,nS]
    rmse = np.sqrt(np.nanmean((pred-obs)**2, axis=0))
    return rmse


def calCorr(pred, obs):
    mask = np.isnan(obs) | np.isnan(pred)
    A = pred.copy()
    B = obs.copy()
    A[mask] = np.nan
    B[mask] = np.nan
    mA = A - np.nanmean(A, axis=0)
    mB = B - np.nanmean(B, axis=0)
    p1 = np.nansum(mA*mB, axis=0)
    p2 = np.sqrt(np.nansum(mA**2, axis=0)*np.nansum(mB**2, axis=0))
    return p1/p2


def calSMAPE(pred, obs):
    mask = np.isnan(obs) | np.isnan(pred)
    A = pred.copy()
    B = obs.copy()
    A[mask] = np.nan
    B[mask] = np.nan
    e = 2*np.abs(A-B)/(np.abs(A)+np.abs(B))
    mape = np.nanmean(e, axis=0)
    return mape


def calMAPE(pred, obs):
    mask = np.isnan(obs) | np.isnan(pred) | (obs == 0)
    A = pred.copy()
    B = obs.copy()
    A[mask] = np.nan
    B[mask] = np.nan
    e = np.abs(A-B)/np.abs(A)
    mape = np.nanmean(e, axis=0)
    return mape

def calLogRMSE(pred, obs):  
    mask = np.isnan(obs) | np.isnan(pred) 
    error=np.sqrt(np.nanmean((np.log(obs/pred))**2,axis=0))    
    return error



def calCorrOld(pred, obs):
    # data in [nT,nS]
    bV = pred.ndim == 1
    if bV:
        pred = pred[:, None]
        obs = obs[:, None]
    [nT, nS] = pred.shape
    corr = np.full([nS], np.nan)
    for k in range(nS):
        a = pred[:, k]
        b = obs[:, k]
        indV = np.where(~np.isnan(a) & ~np.isnan(b))[0]
        with warnings.catch_warnings():
            if np.nanmin(b) == np.nanmax(b):
                print('WARNING connstant observation in calculating corrcoef')
                warnings.simplefilter('ignore', category=RuntimeWarning)
            corr[k] = np.corrcoef(a[indV], b[indV])[0, 1]
    if bV:
        return corr[0]
    else:
        return corr


def calBias(pred, obs):
    # data in [nT,nS,nC]
    bias = np.nanmean(np.abs(pred-obs), axis=0)
    return bias


def calBiasR(pred, obs):
    u1 = np.nanmean(pred, axis=0)
    u2 = np.nanmean(obs, axis=0)
    return u1/u2


def calVarR(pred, obs):
    s1 = np.nanstd(pred, axis=0)
    s2 = np.nanstd(obs, axis=0)
    return s1/s2


def calPercent(x, p, rank=True):
    if rank:
        return np.nanpercentile(x, p*100)
    else:
        vmin = np.nanmin(x)
        vmax = np.nanmax(x)
        return (vmax-vmin)*p+vmin


def gridCorrT(gridX, gridY):
    # grid in [...,nt]
    nt = gridX.shape[-1]
    xm = np.nanmean(gridX, axis=-1)
    ym = np.nanmean(gridY, axis=-1)
    gridXM = np.repeat(xm[..., np.newaxis], nt, axis=-1)
    gridYM = np.repeat(ym[..., np.newaxis], nt, axis=-1)
    dX = gridX-gridXM
    dY = gridY-gridYM
    cov = np.nansum(dX*dY, axis=-1)
    sX = np.sqrt(np.nansum(dX*dX, axis=-1))
    sY = np.sqrt(np.nansum(dY*dY, axis=-1))
    r = cov/(sX*sY)
    return r
