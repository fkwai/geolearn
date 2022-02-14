
import matplotlib.gridspec as gridspec
from hydroDL.model.waterNet import convTS, sepPar
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL.master import basinFull
from hydroDL.model import trainBasin, crit
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
from hydroDL import utils
import torch.nn.functional as F

from hydroDL.model import waterNetTestC

import pandas as pd
import os

dataName = 'B5Y09-00955'
DF = dbBasin.DataFrameBasin(dataName)
codeLst = ['00955']
nc = len(codeLst)
label = 'test'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip' for k in range(nc+1)]
varXC = gageII.varLstEx
mtdXC = ['QT' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()

# model
nh = 16
nr = 5
model = waterNetTestC.Wn0110C2(nh, len(varXC), nr, nc=nc)
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTemp'
modelFile = 'wn0110C2-00955-{}-ep{}'.format(dataName, 1000)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))
model = model.cuda()
optim = torch.optim.Adam(model.parameters())
lossFun = crit.LogLoss2D().cuda()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
sizeLst = trainBasin.getSize(dataTup1)
[rho, nbatch] = batchSize
nIterEp = int(np.ceil((ns*nt)/(nbatch*rho)))
saveDir = r'/scratch/users/kuaifang/temp/'
lossLst = list()

# water net
model.eval()
[x, xc, y, yc] = dataTup2
xP = torch.from_numpy(x).float().cuda()
xcP = torch.from_numpy(xc).float().cuda()
nt, ns, _ = y.shape
t = DF.getT(testSet)
testBatch = 100
iS = np.arange(0, ns, testBatch)
iE = np.append(iS[1:], ns)
qP = np.ndarray([nt-nr+1, ns])
cP = np.ndarray([nt-nr+1, ns])
for k in range(len(iS)):
    print('batch {}'.format(k))
    qOut, cOut = model(xP[:, iS[k]:iE[k], :], xcP[iS[k]:iE[k]])
    qP[:, iS[k]:iE[k]] = qOut.detach().cpu().numpy()
    cP[:, iS[k]:iE[k]] = cOut[:, :, 0].detach().cpu().numpy()
model.zero_grad()
nashQ1 = utils.stat.calNash(qP, y[nr-1:, :, 0])
corrQ1 = utils.stat.calCorr(qP, y[nr-1:, :, 0])
nashC1 = utils.stat.calNash(cP, y[nr-1:, :, 1])
corrC1 = utils.stat.calCorr(cP, y[nr-1:, :, 1])
# LSTM
outName = 'B5Y09-00955-QC'
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=False, ep=1000)
qL = yL[:, :, 0]
cL = yL[:, :, 1]
nashQ2 = utils.stat.calNash(qL, y[:, :, 0])
corrQ2 = utils.stat.calCorr(qL, y[:, :, 0])
nashC2 = utils.stat.calNash(cL, y[:, :, 1])
corrC2 = utils.stat.calCorr(cL, y[:, :, 1])

# steps
x = xP
xc = xcP
nt = x.shape[0]
P, E, T1, T2, R, LAI = [x[:, :, k] for k in range(x.shape[-1])]
nt, ns = P.shape
nh = model.nh
nr = model.nr
nc = model.nc
Sf = torch.zeros(ns, nh).cuda()
Ss = torch.zeros(ns, nh).cuda()
Sg = torch.zeros(ns, nh).cuda()
w = model.fcW(xc)
[kp, ks, kg, gp, gL, qb, ga] = sepPar(w, nh, model.wLst)
gL = gL*2
qb = qb+1e-5
# kg = kg/10
ga = torch.softmax(model.DP(ga), dim=1)
# ga = torch.softmax(ga, dim=1)
xT = torch.cat([x, torch.tile(xc, [nt, 1, 1])], dim=-1)
v = model.fcT(xT)
[vi, ve, vm] = sepPar(v, nh, model.vLst)
vi = F.hardsigmoid(v[:, :, :nh]*2)
ve = ve*2
wc = model.fcC(xc)
[eqs, eqg] = sepPar(wc, nh*nc, model.cLst)
eqs = eqs.view(ns, nh, nc).permute(-1, 0, 1)*10
eqg = eqg.view(ns, nh, nc).permute(-1, 0, 1)*10
Cs = torch.zeros(nc, ns, nh).cuda()
Cg = eqg*qb/kg
xTC = torch.cat([T1[:, :, None], T2[:, :, None],
                torch.tile(xc, [nt, 1, 1])], dim=-1)
vc = model.fcCT(xTC)
[rs, rg] = sepPar(vc, nh*nc, model.cLst)
rs = rs.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
rg = rg.view(nt, ns, nh, nc).permute(-1, 0, 1, 2)
wR = model.fcR(xc)
vf = torch.arccos((T1+T2)/(T2-T1))/3.1415
vf[T1 >= 0] = 0
vf[T2 <= 0] = 1
Ps = P*vf
Pla = P*(1-vf)
Pl = Pla[:, :, None]*vi
Ev = E[:, :, None]*ve
Q1T = torch.zeros(nt, ns, nh).cuda()
Q2T = torch.zeros(nt, ns, nh).cuda()
Q3T = torch.zeros(nt, ns, nh).cuda()
C2T = torch.zeros(nc, nt, ns, nh).cuda()
C3T = torch.zeros(nc, nt, ns, nh).cuda()
for k in range(nt):
    qf = torch.minimum(Sf+Ps[k, :, None], vm[k, :, :])
    Sf = torch.relu(Sf+Ps[k, :, None]-vm[k, :, :])
    H = torch.relu(Ss+Pl[k, :, :]+qf-Ev[k, :, :])
    qp = torch.relu(kp*(H-gL))
    qs = ks*torch.minimum(H, gL)
    Ss = H-qp-qs
    Cs = (Cs+rs[:, k, :, :]*eqs*Ss)/(1+ks+rs[:, k, :, :])
    qsg = qs*gp
    qg = kg*(Sg+qsg)+qb
    Sg = (1-kg)*(Sg+qsg)-qb
    Hg = Sg+qb/kg
    Cg = (Cg+rg[:, k, :, :]*eqg*Hg+Cs*ks*gp)/(1+kg+rg[:, k, :, :])
    Q1T[k, :, :] = qp
    Q2T[k, :, :] = qs*(1-gp)
    Q3T[k, :, :] = qg
    C2T[:, k, :, :] = Cs*ks*(1-gp)
    C3T[:, k, :, :] = Cg*kg
r = torch.relu(wR[:, :nh*nr])
Q1R = convTS(Q1T, r)
Q2R = convTS(Q2T, r)
Q3R = convTS(Q3T, r)
outQ = torch.sum((Q1R+Q2R+Q3R)*ga, dim=2)
outCLst = list()
for k in range(nc):
    C2R = convTS(C2T[k, ...], r)
    C3R = convTS(C3T[k, ...], r)
    temp = torch.sum((C2R+C3R)*ga, dim=2)/outQ
    outCLst.append(temp)
outC = torch.stack(outCLst, dim=-1)

# plot some variables
outEqg = torch.sum(eqg*ga, dim=2).detach().cpu().numpy()[0,:]
outEqs = torch.sum(eqs*ga, dim=2).detach().cpu().numpy()[0,:]
lat, lon = DF.getGeo()

figM = plt.figure()
gsM = gridspec.GridSpec(2, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, outEqs, vRange=[5, 15])
axM0.set_title('Ceq soil')
axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, outEqg, vRange=[5, 15])
axM1.set_title('Ceq gw')
figM.show()
