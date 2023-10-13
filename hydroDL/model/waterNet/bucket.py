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
    # qp = torch.relu(H - L)
    qs = k2 * torch.minimum(H, L)
    Ss = H - qp - qs
    return Ss, qp, qs


def deep(S, I, *, k, baseflow):
    qd = k * (S + I) + baseflow
    Sd = (1 - k) * (S + I) - baseflow
    return (Sd,)


def expBucket(D, I, *, gk, gl):
    gk = gk.unsqueeze(-1) + 1e-5
    gl = gl.unsqueeze(-1)
    ns, nh, nd = D.shape
    D_new = D.clone()
    D_new[:, :, :-1] = D[:, :, 1:]
    D_total = torch.relu(D[:, :, -1] + I)
    D_total_mat = D_total.unsqueeze(-1).repeat(1, 1, nd)
    D_new[:, :, -1] = D_total
    D_update = torch.minimum(D_new, D_total_mat)
    # k1 = torch.exp(gk * (D_update - gl)) / gk
    # k1[D_update > gl] = D_update[D_update > gl]
    # k2 = torch.cat([torch.exp(gk * -gl) / gk, k1[:, :, :-1]], dim=-1)
    # Q = torch.relu((k1 - k2))  # very small rounding error

    # test for tanh
    k1 = torch.log(torch.cosh(D_update / gl))
    # k2 = torch.cat([, k1[:, :, :-1]], dim=-1)


    # k1 = torch.exp(gk * (D_update - gl))
    # k1[D_update > gl] = D_update[D_update > gl]
    # k2 = torch.cat([torch.exp(gk * -gl), k1[:, :, :-1]], dim=-1)
    # Q = torch.relu((k1 - k2) / gk)  # very small rounding error

    Q_cum = torch.cumsum(Q, -1)
    D_update = D_update - Q_cum
    return D_update, Q


def step(iT, S, I, param):
    Sf, Ss, Sd = S
    Pl, Ps, Evp = I
    paramK, paramG, paramR = param
    Sf_new, qf = snow(Sf, Ps[iT, ...], paramK['km'][iT, ...])
    # interception and evaporation factor
    if 'gi' in paramG:
        P = Pl[iT, ...] * paramG['gi']
    elif 'ki' in paramK:
        P = Pl[iT, ...] * paramG['ki'][iT, ...]
    if 'ge' in paramG:
        E = Evp[iT, ...] * paramG['ge']
    elif 'ke' in paramK:
        E = Evp[iT, ...] * paramG['ke'][iT, ...]
    Is = qf + P - E
    Ss_new, qp, qsA = shallow(Ss, Is, L=paramG['gl'], k1=paramG['kp'], k2=paramG['ks'])
    qs = qsA * (1 - paramG['gd'])
    Id = qsA * paramG['gd']
    Sd_new, qd = deep(Sd, Id, k=paramG['kd'], baseflow=paramG['qb'])
    return (Sf_new, Ss_new, Sd_new), (qf, qp, qs, qd)


def stepSAS(iT, S, I, param):
    Sf, D = S
    Pl, Ps, Evp = I
    paramK, paramG = param
    Sf_new, qf = snow(Sf, Ps[iT, ...], paramK['km'][iT, ...])
    # interception and evaporation factor
    if 'gi' in paramG:
        P = Pl[iT, ...] * paramG['gi']
    elif 'ki' in paramK:
        P = Pl[iT, ...] * paramG['ki'][iT, ...]
    if 'ge' in paramG:
        E = Evp[iT, ...] * paramG['ge']
    elif 'ke' in paramK:
        E = Evp[iT, ...] * paramG['ke'][iT, ...]
    Is = qf + P - E
    D_new, Q = expBucket(D, Is, gk=paramG['gk'], gl=paramG['gl'])
    return (Sf_new, D_new), (qf, Q)
