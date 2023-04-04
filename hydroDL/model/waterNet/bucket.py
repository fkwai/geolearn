"""
define bucket functions of waterNet
input: State (S), Source term (I), Parameters
"""

import torch


def divideP(P, Tmin, Tmax):
    vf = torch.arccos((Tmin + Tmax) / (Tmax - Tmin)) / 3.1415926
    vf[Tmin >= 0] = 0
    vf[Tmax <= 0] = 1
    Ps = P * vf
    Pl = P * (1 - vf)
    return Ps, Pl


def snow(S, snow_fall, snow_melt):
    qf = torch.minimum(S + snow_fall, snow_melt)
    Sf = torch.relu(S + snow_fall - snow_melt)
    return Sf, qf


def shallow(S, I, *, L, k1, k2):
    H = torch.relu(S + I)
    # qp = torch.relu(k1 * (H - L))
    qp = torch.relu(H - L)
    qs = k2 * torch.minimum(H, L)
    Ss = H - qp - qs
    return Ss, qp, qs


def deep(S, I, *, k, baseflow):
    qd = k * (S + I) + baseflow
    Sd = (1 - k) * (S + I) - baseflow
    return Sd, qd


def step(iT, S, I, param):
    Sf, Ss, Sd = S
    Pl, Ps, Evp = I
    paramK, paramG, paramR = param
    Sf_new, qf = snow(Sf, Ps[iT, ...], paramK['km'][iT, ...])
    Is = qf + Pl[iT, ...] * paramG['gi'] - Evp[iT, ...] * paramG['ge']
    Ss_new, qp, qsA = shallow(
        Ss, Is, L=paramG['gl'], k1=paramG['kp'], k2=paramG['ks']
    )
    qs = qsA * (1 - paramG['gd'])
    Id = qsA * paramG['gd']
    Sd_new, qd = deep(Sd, Id, k=paramG['kd'], baseflow=paramG['qb'])
    return (Sf_new, Ss_new, Sd_new), (qf, qp, qs, qd)
