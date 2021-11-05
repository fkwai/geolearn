import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


def waterForward(x, w, v, nh, outQ=False):
    P, E, T1, T2, LAI = [x[:, :, k] for k in range(5)]
    nt, ns = P.shape
    Ta = (T1+T2)/2
    rP = 1-torch.arccos((T1+T2)/(T2-T1))/3.1415
    rP[T1 >= 0] = 1
    rP[T2 <= 0] = 0
    Ps = (1-rP)*P
    Pl = rP*P
    S0 = torch.zeros(ns, nh).cuda()
    # S1 = torch.zeros(ns, self.nh).cuda()
    S2 = torch.zeros(ns, nh).cuda()
    S3 = torch.zeros(ns, nh).cuda()
    Yout = torch.zeros(nt, ns).cuda()
    gm = torch.exp(w[:, :nh])+1
    ge = torch.sigmoid(w[:, nh:nh*2])*2
    k2 = torch.sigmoid(w[:, nh*2:nh*3])
    k23 = torch.sigmoid(w[:, nh*3:nh*4])
    k3 = torch.sigmoid(w[:, nh*4:nh*5])/10
    gl = torch.exp(w[:, nh*5:nh*6])*2
    ga = torch.softmax(nn.Dropout(w[:, nh*6:nh*7]), dim=1)
    qb = torch.relu(w[:, nh*7:nh*8])
    vi = torch.sigmoid(v[:, :, :nh])
    if outQ:
        Q1out = torch.zeros(nt, ns, nh).cuda()
        Q2out = torch.zeros(nt, ns, nh).cuda()
        Q3out = torch.zeros(nt, ns, nh).cuda()

    for k in range(nt):
        H0 = S0+Ps[k, :, None]
        qSm = torch.minimum(H0, torch.relu(Ta[k, :, None]*gm))
        qIn = qSm+Pl[k, :, None]*vi[k, :, :] - E[k, :, None]*ge
        Q1 = torch.relu(S2+qIn-gl)
        H2 = torch.minimum(torch.relu(S2+qIn), gl)
        q2 = H2*k2
        Q2 = q2*(1-k23)
        H3 = S3+q2*k23
        Q3 = torch.relu(H3*k3+qb)
        S0 = H0-qSm
        S2 = H2-Q2
        S3 = H3-Q3
        Y = torch.sum((Q1+Q2+Q3)*ga, dim=1)
        Yout[k, :] = Y
        if outQ:
            Q1out[k, :, :] = Q1
            Q2out[k, :, :] = Q2
            Q3out[k, :, :] = Q3
