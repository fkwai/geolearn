from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet

# test case
siteNo = '07241550'
siteNo = '06752260'
df = dbBasin.readSiteTS(
    siteNo,  ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr', 'runoff'])
P = df['pr'].values
Q = df['runoff'].values/365*1000
E = df['pet'].values
# T = (df['tmmn'].values+df['tmmx'].values)/2 - 273.15
T = df['tmmn'].values - 273.15
t = df.index.values.astype('datetime64[D]')

# fig, ax = plt.subplots(1, 1)
# ax.plot(df['srad'], 'r')
# ax.twinx().plot(df['pr'])
# fig.show()


# fig, ax = plt.subplots(1, 1)
# ax.plot(df['pet'], 'r')
# ax.twinx().plot(df['pr'])
# fig.show()


importlib.reload(waterNet)
nh = 16
nbatch = 100
rho = 365
nt = len(P)

# nt1 = np.where(df.index.values.astype('datetime64[D]'))
nt1 = 12000
model = waterNet.WaterNetModel2(nh, nm=0)
model = model.cuda()
optim = torch.optim.Adagrad(model.parameters(), lr=1)
# optim = torch.optim.Adadelta(model.parameters(), lr=1)

lossFun = torch.nn.MSELoss().cuda()
model.state_dict()
model.train()
for kk in range(100):
    iT = np.random.randint(0, nt1-rho, [nbatch])
    pTemp = np.full([rho, nbatch], np.nan)
    qTemp = np.full([rho, nbatch], np.nan)
    tTemp = np.full([rho, nbatch], np.nan)
    eTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        pTemp[:, k] = P[iT[k]+1:iT[k]+rho+1]
        qTemp[:, k] = Q[iT[k]+1:iT[k]+rho+1]
        tTemp[:, k] = T[iT[k]+1:iT[k]+rho+1]
        eTemp[:, k] = E[iT[k]+1:iT[k]+rho+1]
    pT = torch.from_numpy(pTemp).float().cuda()
    qT = torch.from_numpy(qTemp).float().cuda()
    tT = torch.from_numpy(tTemp).float().cuda()
    eT = torch.from_numpy(eTemp).float().cuda()
    model.zero_grad()
    qP, fP, hP, gP = model(pT, tT, eT)
    loss = lossFun(qP, qT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)
    print(model.state_dict()['SN.w'])


model.eval()
qT = torch.from_numpy(Q[nt1:, None]).float().cuda()
pT = torch.from_numpy(P[nt1:, None]).float().cuda()
tT = torch.from_numpy(T[nt1:, None]).float().cuda()
eT = torch.from_numpy(T[nt1:, None]).float().cuda()

qP, hP, sP = model(pT, tT)
model.zero_grad()
loss = lossFun(qP, qT)
yP = qP[:, 0].detach().cpu().numpy()
hOut = hP[:, 0].detach().cpu().numpy()
sOut = sP[:, 0].detach().cpu().numpy()


tt = df.index.values[nt1:]
fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, P[nt1:])
axes[0].plot(tt, yP, '-r')
axes[1].plot(tt, Q[nt1:])
axes[1].plot(tt, yP, '-r')
fig.show()

fig, axes = plt.subplots(3, 1)
axes[0].plot(df.index.values, P)
axes[1].plot(df.index.values, Q)
axes[2].plot(df.index.values, T)
axes[2].plot(df.index.values, Q)
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, hOut)
axes[1].plot(tt, sOut)
fig.show()


dictPar = model.state_dict()
dictPar.keys()
torch.sigmoid(dictPar['w_i'])
torch.softmax(dictPar['w_o'], dim=0)

torch.sum(torch.softmax(dictPar['w_o'], dim=0))


torch.sigmoid(dictPar['LN.w'])
torch.exp(dictPar['SN.w'])

utils.stat.calNash(yP, Q[nt1:])

torch.pow(torch.tensor([-1, 0, 1, 2]), torch.tensor([-1, 0, 1, 2]))
torch.exp(torch.tensor([-1, 0, 1, 2]))
