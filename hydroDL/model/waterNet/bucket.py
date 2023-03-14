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


def snow(S, Ps, fm):
    qf = torch.minimum(S + Ps, fm)
    Sf = torch.relu(S + Ps - fm)
    return Sf, qf


def shallow(S, I, param):
    [gL, kp, ks] = param
    H = torch.relu(S + I)
    qp = torch.relu(kp * (H - gL))
    qs = ks * torch.minimum(H, gL)
    Ss = H - qp - qs
    return Ss, qp, qs


def deep(S, I, param):
    [kd, qb] = param
    qd = kd * (S + I) + qb
    Sd = (1 - kd) * (S + I) - qb
    return Sd, qd

