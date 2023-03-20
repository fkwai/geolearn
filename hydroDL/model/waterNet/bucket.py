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
    qp = torch.relu(k1 * (H - L))
    qs = k2 * torch.minimum(H, L)
    Ss = H - qp - qs
    return Ss, qp, qs


def deep(S, I, *, k, baseflow):
    qd = k * (S + I) + baseflow
    Sd = (1 - k) * (S + I) - baseflow
    return Sd, qd
