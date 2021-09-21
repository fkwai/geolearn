import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet, crit
from hydroDL.model import rnn

# test case
siteNo = '06752260'
# siteNo = '06752260'
# siteNo = '11264500'

df = dbBasin.readSiteTS(siteNo, ['pr', 'runoff', 'pet', 'tmmn', 'tmmx'])
P = df['pr'].values
Q = df['runoff'].values
E = df['pet'].values
T = df['tmmn'].values
t = df.index.values

p = (P-np.nanmin(P))/(np.nanpercentile(P, 90)-np.nanmin(P))
q = (Q-np.nanmin(Q))/(np.nanpercentile(Q, 90)-np.nanmin(Q))
e = (E-np.nanmin(E))/(np.nanpercentile(E, 90)-np.nanmin(E))
f = (E-np.nanmin(T))/(np.nanmax(T)-np.nanmin(T))

# sn = 1e-5
# logp = np.log(P+sn)
# logq = np.log(Q+sn)
# p = (logp-np.nanmin(logp))/(np.nanmax(logp)-np.nanmin(logp))
# q = (logq-np.nanmin(logq))/(np.nanmax(logq)-np.nanmin(logq))

# fig, ax = plt.subplots(1, 1)
# # ax.hist(p[P != 0], bins=100)
# ax.hist(Q[Q != 0], bins=100)
# fig.show()

p = np.stack([p, e]).T


importlib.reload(waterNet)
nh = 32
nbatch = 50
rho = 365
nt = len(P)
nt1 = 12000
# model = waterNet.RnnModel(nh)
model = rnn.LstmModel(nx=2, ny=1, hiddenSize=10)
model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())
lossFun = torch.nn.MSELoss().cuda()

model.train()
for kk in range(500):
    iT = np.random.randint(0, nt1-rho, [nbatch])
    xTemp = np.full([rho, nbatch, p.shape[1]], np.nan)
    yTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        xTemp[:, k, :] = p[iT[k]+1:iT[k]+rho+1]
        yTemp[:, k] = q[iT[k]+1:iT[k]+rho+1]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    model.zero_grad()
    yP = model(xT)
    loss = lossFun(yP, yT[:, :, None])
    optim.zero_grad()

    loss.backward()
    optim.step()
    print(loss)


model.eval()
xT = torch.from_numpy(p[nt1:, None, :]).float().cuda()
yT = torch.from_numpy(q[nt1:, None]).float().cuda()
yOut = model(xT)
model.zero_grad()
loss = lossFun(yOut, yT[:, :, None])
yP = yOut[:, 0].detach().cpu().numpy()

tt = t[nt1:]
fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, yP, '-r')
axes[0].plot(tt, p[nt1:])
axes[1].plot(tt, yP, '-r')
axes[1].plot(tt, q[nt1:])
fig.show()



model.eval()
xT = torch.from_numpy(p[:1000, None]).float().cuda()
yT = torch.from_numpy(q[:1000, None]).float().cuda()
yOut = model(xT)
model.zero_grad()
loss = lossFun(yOut, yT[:, :, None])
yP = yOut[:, 0].detach().cpu().numpy()

tt = t[:1000]
fig, axes = plt.subplots(2, 1)
# axes[0].plot(tt,yP[:1000], '-r')
axes[0].plot(tt, p[:1000, 0])
axes[1].plot(tt, yP[:1000], '-r')
axes[1].plot(tt, q[:1000])
fig.show()