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
df = dbBasin.readSiteTS(siteNo,  varLst)
P = df['pr'].values
Q = df['runoff'].values/365*1000
E = df['pet'].values
# T = (df['tmmn'].values+df['tmmx'].values)/2 - 273.15
T = df['tmmn'].values - 273.15
C = df['00915'].values
# C = (C-np.nanmin(C))/(np.nanpercentile(C, 90)-np.nanmin(C))

t = df.index.values.astype('datetime64[D]')

fig, ax = plt.subplots(1, 1)
ax.plot(Q, 'b')
ax.twinx().plot(C, '*r')
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(Q, 'b')
ax.twinx().plot(Q*C, '*r')
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(df['pet'], 'r')
ax.twinx().plot(df['etr'])
fig.show()


importlib.reload(waterNet)
nh = 4
nbatch = 100
rho = 365
nt = len(P)

# nt1 = np.where(df.index.values.astype('datetime64[D]'))
nt1 = 12000
model = waterNet.SoilModel(nh, nm=0)
model = model.cuda()
optim = torch.optim.Adagrad(model.parameters(), lr=1)
optim = torch.optim.RMSprop(model.parameters(), lr=1)
# optim = torch.optim.Rprop(model.parameters(), lr=0.1)
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
    qP, cP, _ = model(pT, tT, eT)
    loss1 = lossFun(qP, qT)
    loss2 = lossFun(cP, cT)
    loss = loss1*loss2
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(kk, loss1.item(), loss2.item())


model.eval()
qT = torch.from_numpy(Q[nt1:, None]).float().cuda()
pT = torch.from_numpy(P[nt1:, None]).float().cuda()
tT = torch.from_numpy(T[nt1:, None]).float().cuda()
eT = torch.from_numpy(E[nt1:, None]).float().cuda()

qP, cP, (sP, h1P, h2P, qsP) = model(pT, tT, eT)
model.zero_grad()
loss = lossFun(qP, qT)
qOut = qP[:, 0].detach().cpu().numpy()
sOut = sP[:, 0].detach().cpu().numpy()
h1Out = h1P[:, 0].detach().cpu().numpy()
h2Out = h2P[:, 0].detach().cpu().numpy()
cOut = cP[:, 0].detach().cpu().numpy()
qsOut = qsP[:, 0].detach().cpu().numpy()


tt = df.index.values[nt1:]
fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, C[nt1:], '*b')
axes[0].plot(tt, cOut, '-r')
axes[1].plot(tt, Q[nt1:], '-b')
axes[1].plot(tt, qOut, '-r')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(df.index.values, P)
axes[0].twinx().plot(df.index.values, Q, 'k')
axes[1].plot(df.index.values, T)
axes[1].twinx().plot(df.index.values, Q, 'k')
fig.show()

fig, axes = plt.subplots(3, 1)
axes[0].plot(tt, sOut)
axes[1].plot(tt, h1Out)
axes[2].plot(tt, h2Out)
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(tt, qsOut)
fig.show()


dictPar = model.state_dict()
dictPar.keys()

torch.sigmoid(dictPar['w_i'])
torch.softmax(dictPar['w_o'], dim=0)

torch.sum(torch.softmax(dictPar['w_o'], dim=0))

torch.exp(model.Pond.wl*2)
utils.stat.calNash(yP, Q[nt1:])

a = torch.softmax(dictPar['wo'], dim=0).detach().cpu().numpy()
we = torch.sigmoid(dictPar['Soil.we']).detach().cpu().numpy()
wt = torch.sigmoid(dictPar['Soil.wt']).detach().cpu().numpy()
wk = torch.sigmoid(dictPar['Soil.wk']).detach().cpu().numpy()
wl = torch.exp(dictPar['Soil.wl']*3).detach().cpu().numpy()
b = torch.relu(dictPar['b']).detach().cpu().numpy()

q = hOut*wk+b
qr = hOut*wk+b

c = np.log(1/wk)
cP = np.sum(q*c*a, axis=1)/(yP+0.01)


fig, axes = plt.subplots(2, 1)
axes[0].plot(df['00915'], '*r')
axes[0].twinx().plot(df['runoff'])
# axes[1].plot(np.log(df['runoff'].values), df['00915'].values, '*')
axes[1].plot(df['runoff'].values, df['00915'].values, '*')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, cP, '*r')
axes[0].twinx().plot(tt, df['00915'].values[nt1:], '*')
axes[1].plot(yP, cP, '*r')
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(df['00915'].values[nt1:], cP, '*r')
axes[1].plot(np.log(df['runoff'].values[nt1:]), cP, '*')
fig.show()
