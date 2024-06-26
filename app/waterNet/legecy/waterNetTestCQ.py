from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin, gridMET
import torch
import importlib
from hydroDL.model import waterNet, crit

# test case
siteNo = '07241550'
siteNo = '06752260'
varLst = gridMET.varLst + ['runoff']+['00915']
df = dbBasin.readSiteTS(siteNo, varLst)
P = df['pr'].values
Q = df['runoff'].values/365*1000
E = df['pet'].values
# T = (df['tmmn'].values+df['tmmx'].values)/2 - 273.15
T = df['tmmn'].values - 273.15
C = df['00915'].values
C = (C-np.nanmin(C))/(np.nanpercentile(C,90)-np.nanmin(C))
t = df.index.values.astype('datetime64[D]')

fig, ax = plt.subplots(1, 1)
ax.plot(df['srad'], 'r')
ax.twinx().plot(df['pr'])
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(df['pet'], 'r')
ax.twinx().plot(df['pr'])
fig.show()


importlib.reload(waterNet)
nh = 16
nbatch = 100
rho = 365
nt = len(P)

# nt1 = np.where(df.index.values.astype('datetime64[D]'))
nt1 = 12000
model = waterNet.WaterNetModel(nh, nm=0)
model = model.cuda()
optim = torch.optim.Adagrad(model.parameters(), lr=1)
# optim = torch.optim.Adadelta(model.parameters(), lr=1)

# lossFun = torch.nn.MSELoss().cuda()
lossFun = crit.NSELoss2D().cuda()
model.state_dict()
model.train()
for kk in range(100):
    iT = np.random.randint(0, nt1-rho, [nbatch])
    pTemp = np.full([rho, nbatch], np.nan)
    qTemp = np.full([rho, nbatch], np.nan)
    tTemp = np.full([rho, nbatch], np.nan)
    eTemp = np.full([rho, nbatch], np.nan)
    cTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        pTemp[:, k] = P[iT[k]+1:iT[k]+rho+1]
        qTemp[:, k] = Q[iT[k]+1:iT[k]+rho+1]
        tTemp[:, k] = T[iT[k]+1:iT[k]+rho+1]
        eTemp[:, k] = E[iT[k]+1:iT[k]+rho+1]
        cTemp[:, k] = C[iT[k]+1:iT[k]+rho+1]
    pT = torch.from_numpy(pTemp).float().cuda()
    qT = torch.from_numpy(qTemp).float().cuda()
    tT = torch.from_numpy(tTemp).float().cuda()
    eT = torch.from_numpy(eTemp).float().cuda()
    cT = torch.from_numpy(cTemp).float().cuda()

    model.zero_grad()
    qP, cP, hP, sP = model(pT, tT)
    loss1 = lossFun(qP, qT)
    loss2 = lossFun(cP, cT)
    loss = loss1*loss2
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss1.item(), loss2.item())
    # print(model.state_dict()['SN.w'])


model.eval()
qT = torch.from_numpy(Q[nt1:, None]).float().cuda()
pT = torch.from_numpy(P[nt1:, None]).float().cuda()
tT = torch.from_numpy(T[nt1:, None]).float().cuda()
eT = torch.from_numpy(E[nt1:, None]).float().cuda()
cT = torch.from_numpy(C[nt1:, None]).float().cuda()

qP, cP, hP, sP = model(pT, tT)
model.zero_grad()
loss = lossFun(qP, qT)
loss = lossFun(cP, cT)

qOut = qP[:, 0].detach().cpu().numpy()
cOut = cP[:, 0].detach().cpu().numpy()
hOut = hP[:, 0].detach().cpu().numpy()
sOut = sP[:, 0].detach().cpu().numpy()


tt = df.index.values[nt1:]
fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, C[nt1:], '*b')
axes[0].plot(tt, cOut, '-r')
axes[1].plot(tt, Q[nt1:], '-b')
axes[1].plot(tt, qOut, '-r')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, hOut)
axes[1].plot(tt, sOut)
fig.show()

fig, ax = plt.subplots(1, 1)
# ax.plot(np.log(qOut+1e-5), cOut, '*')
ax.plot(qOut, cOut, '*')
fig.show()


dictPar = model.state_dict()
dictPar.keys()
fMat = torch.sigmoid(dictPar['w_i']).detach().cpu().numpy()
kMat = torch.sigmoid(dictPar['LN.wk']).detach().cpu().numpy()
aMat = torch.sigmoid(dictPar['w_o']).detach().cpu().numpy()
tc = 1+torch.exp(dictPar['w_c']).detach().cpu().numpy()

# rMat = np.exp(1/kMat)
# rMat = np.log(1/kMat)
rMat = 1/kMat/tc
# rMat[rMat > 1] = 1

n = 2975
c = np.ndarray(n)
Y = np.ndarray(n)
for k in range(1, n):
    q = hOut[k-1, :]*kMat
    Y[k] = np.mean(q*aMat)
    c[k] = np.mean(q*aMat*rMat+1e-5)/(np.mean(q*aMat)+1e-5)

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.log(Y+1e-5), c, '*')
axes[1].plot(np.log(Q+1e-5), C, '*')
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(Q, 'b-')
ax.twinx().plot(C, 'r*')
fig.show()
