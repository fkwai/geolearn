import torch
import torch.nn.functional as F


def divideP(P, Tmin, Tmax):
    vf = torch.arccos((Tmin + Tmax) / (Tmax - Tmin)) / 3.1415926
    vf[Tmin >= 0] = 0
    vf[Tmax <= 0] = 1
    Ps = P * vf
    Pl = P * (1 - vf)
    return Ps, Pl


def convTS(x, w):
    nt, ns, nh = x.shape
    if w.dim() == 1:
        w = w.repeat([ns, 1])
    nr = int(w.shape[-1] / nh)
    r = torch.softmax(w.view(ns * nh, 1, nr), dim=-1)
    a = x.permute(1, 2, 0).view(1, ns * nh, nt)
    y = F.conv1d(a, r, groups=ns * nh).view(ns, nh, nt - nr + 1).permute(2, 0, 1)
    return y


def sepPar(p, nh, actLst):
    outLst = list()
    for k, act in enumerate(actLst):
        if act == 'skip':
            outLst.append(p[..., nh * k : nh * (k + 1)])
        else:
            if hasattr(torch, act):
                ff = getattr(torch, act)
            elif hasattr(F, act):
                ff = getattr(F, act)
            else:
                Exception('can not find activate func')
            outLst.append(ff(p[..., nh * k : nh * (k + 1)]))
    return outLst


def sepParam(p, nh, pDict, raw=False):
    outDict = dict()
    # pp = p.view(p.shape[0],nh,-1)
    for k, key in enumerate(pDict.keys()):
        ff = pDict[key]
        if ff is None or raw:
            outDict[key] = p[..., nh * k : nh * (k + 1)]
        else:
            outDict[key] = ff(p[..., nh * k : nh * (k + 1)])
    return outDict


def onePeakWeight(w, nh, nr):
    ns = w.shape[0]
    w = torch.sigmoid(w)
    u = torch.relu(w[:, :nh])
    r = torch.cumsum(torch.relu(w[:, nh:].view(ns, nh, nr)), dim=-1)
    k = torch.softmax(r - u[:, :, None] * nr, dim=-1).permute(2, 0, 1)
    return k
