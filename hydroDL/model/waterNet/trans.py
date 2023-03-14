import torch

def divideP(P, Tmin, Tmax):
    vf = torch.arccos((Tmin + Tmax) / (Tmax - Tmin)) / 3.1415926
    vf[Tmin >= 0] = 0
    vf[Tmax <= 0] = 1
    Ps = P * vf
    Pl = P * (1 - vf)
    return Ps, Pl