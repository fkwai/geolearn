import torch
from hydroDL.model.waterNet.func import convTS, sepParam

ns = 10
nh = 4


def fakeQ():
    Qp = torch.rand(ns, nh)
    Qs = torch.rand(ns, nh)
    Qd = torch.rand(ns, nh)
    return Qp, Qs, Qd


nr = 5
w = torch.rand(ns, nh, nr)

t = torch.rand(ns, nh * nr)

pDict = dict(r1=None, r2=None, r3=None, r4=None, r5=None)
outDict = sepParam(t, nh, pDict)

tt = t.view(ns, nr, nh).permute(0, 2, 1)
tt.view(ns*nh,1,nr)

tt[:,:,4]
outDict['r5']
