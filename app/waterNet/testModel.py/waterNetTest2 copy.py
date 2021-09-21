import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet

# test case
# siteNo = '07241550'
siteNo = '06752260'
# siteNo = '11264500'
df = dbBasin.readSiteTS(siteNo, ['pr', 'runoff', 'pet', 'tmmn', 'tmmx'])
P = df['pr'].values
Q = df['runoff'].values/365*1000
E = df['pet'].values
T = (df['tmmn'].values+df['tmmx'].values)/2 - 273.15
# T = df['tmmn'].values - 273.15

# sn = 1e-5
# logp = np.log(P+sn)
# logq = np.log(Q+sn)
# p = (logp-np.nanmin(logp))/(np.nanmax(logp)-np.nanmin(logp))
# q = (logq-np.nanmin(logq))/(np.nanmax(logq)-np.nanmin(logq))

# fig, ax = plt.subplots(1, 1)
# ax.hist(p[P != 0], bins=100)
# # ax.hist(q[Q != 0], bins=100)
# fig.show()

importlib.reload(waterNet)
nh = 16
nbatch = 50
rho = 1000
nt = len(P)
nt1 = 12000
model = waterNet.WaterNetModel(nh, nm=2)
model = model.cuda()
optim = torch.optim.Adagrad(model.parameters(), lr=1)
# optim = torch.optim.Adadelta(model.parameters())

lossFun = torch.nn.MSELoss().cuda()
model.state_dict()
model.train()
for kk in range(200):
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
    qP, hP, sP = model(pT, tT)
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
eT = torch.from_numpy(EOFError[nt1:, None]).float().cuda()

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
axes[0].plot(tt, P)
axes[1].plot(tt, Q)
axes[2].plot(tt, T)
axes[2].plot(tt, Q)
fig.show()

fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, hOut)
axes[1].plot(tt, sOut)
fig.show()

dictPar = model.state_dict()
dictPar.keys()
torch.sigmoid(dictPar['w_i'])
torch.softmax(dictPar['w_o'], dim=0)
torch.sigmoid(dictPar['LN.w'])
torch.exp(dictPar['SN.w'])


x = np.ones([1, 1])
h = np.zeros([1, nh])
