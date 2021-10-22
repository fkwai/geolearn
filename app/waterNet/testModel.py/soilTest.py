from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet, crit

# test case
siteNo = '07241550'
siteNo = '06752260'
df = dbBasin.readSiteTS(
    siteNo,  ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr', 'runoff'])
P = df['pr'].values
Q = df['runoff'].values/365*1000
E = df['etr'].values
# T = (df['tmmn'].values+df['tmmx'].values)/2 - 273.15
T1 = df['tmmn'].values - 273.15
T2 = df['tmmx'].values - 273.15

t = df.index.values.astype('datetime64[D]')

fig, ax = plt.subplots(1, 1)
ax.plot(df['srad'], 'r')
ax.twinx().plot(df['pr'])
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(df['pet'], 'r')
ax.twinx().plot(df['etr'])
fig.show()


importlib.reload(waterNet)
nh = 8
nbatch = 100
rho = 365
nt = len(P)

# nt1 = np.where(df.index.values.astype('datetime64[D]'))
nt1 = 12000
model = waterNet.SoilModel(nh, nm=0)
model = model.cuda()
optim = torch.optim.RMSprop(model.parameters(), lr=0.1)
# optim = torch.optim.Adagrad(model.(), lr=1)

# optim = torch.optim.Adadelta(model.parameters(), lr=1)

lossFun = torch.nn.MSELoss().cuda()
# lossFun = crit.NSELoss2D()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
model.train()
for kk in range(50):
    iT = np.random.randint(0, nt1-rho, [nbatch])
    pTemp = np.full([rho, nbatch], np.nan)
    qTemp = np.full([rho, nbatch], np.nan)
    t1Temp = np.full([rho, nbatch], np.nan)
    t2Temp = np.full([rho, nbatch], np.nan)
    eTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        pTemp[:, k] = P[iT[k]+1:iT[k]+rho+1]
        qTemp[:, k] = Q[iT[k]+1:iT[k]+rho+1]
        t1Temp[:, k] = T1[iT[k]+1:iT[k]+rho+1]
        t2Temp[:, k] = T2[iT[k]+1:iT[k]+rho+1]
        eTemp[:, k] = E[iT[k]+1:iT[k]+rho+1]
    pT = torch.from_numpy(pTemp).float().cuda()
    qT = torch.from_numpy(qTemp).float().cuda()
    t1T = torch.from_numpy(t1Temp).float().cuda()
    t2T = torch.from_numpy(t2Temp).float().cuda()
    eT = torch.from_numpy(eTemp).float().cuda()
    model.zero_grad()
    qP, _ = model(pT, t1T, t2T, eT)
    loss = lossFun(qP, qT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)
    if kk % 10 == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)


model.eval()
qT = torch.from_numpy(Q[nt1:, None]).float().cuda()
pT = torch.from_numpy(P[nt1:, None]).float().cuda()
t1T = torch.from_numpy(T1[nt1:, None]).float().cuda()
t2T = torch.from_numpy(T2[nt1:, None]).float().cuda()
eT = torch.from_numpy(E[nt1:, None]).float().cuda()

qP, (sP, hP, gP, qsP) = model(pT, t1T, t2T, eT)
model.zero_grad()
loss = lossFun(qP, qT)
yP = qP[:, 0].detach().cpu().numpy()
sOut = sP[:, 0].detach().cpu().numpy()
hOut = hP[:, 0].detach().cpu().numpy()
gOut = gP[:, 0].detach().cpu().numpy()
qsOut = qsP[:, 0].detach().cpu().numpy()


tt = df.index.values[nt1:]
fig, axes = plt.subplots(2, 1)
# axes[0].plot(tt, P[nt1:])
axes[0].plot(tt, T1[nt1:], 'b')
axes[0].plot(tt, T2[nt1:], 'g')
axes[0].twinx().plot(tt, yP, '-r')
axes[1].plot(tt, Q[nt1:])
axes[1].plot(tt, yP, '-r')
fig.show()

fig, axes = plt.subplots(3, 1)
axes[0].plot(tt, sOut)
axes[1].plot(tt, hOut)
axes[2].plot(tt, gOut)
fig.show()

fig, ax = plt.subplots(1, 1)
ax.plot(tt, qsOut)
fig.show()


dictPar = model.state_dict()
dictPar.keys()
dictPar['Soil.wk1']
torch.sigmoid(dictPar['Soil.wk1'])
dictPar['Soil.wk2']
torch.sigmoid(dictPar['Soil.wk2'])

torch.exp(dictPar['Soil.wl'])


torch.softmax(dictPar['wo'], dim=0)

torch.sum(torch.softmax(dictPar['w_o'], dim=0))

torch.exp(model.Pond.wl*2)
utils.stat.calNash(yP, Q[nt1:])
