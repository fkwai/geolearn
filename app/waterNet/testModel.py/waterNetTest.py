from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin, gridMET
import torch
import importlib
from hydroDL.model import waterNet

# test case
siteNo = '06623800'
varLst = gridMET.varLst + ['runoff']+['00915']
df = dbBasin.readSiteTS(siteNo, varLst)
P = df['pr'].values
Q = df['runoff'].values
E = df['pet'].values
T1 = df['tmmx'].values - 273.15
T2 = df['tmmn'].values - 273.15
t = df.index.values.astype('datetime64[D]')


importlib.reload(waterNet)
nh = 16
nbatch = 100
rho = 365
nt = len(P)

# nt1 = np.where(df.index.values.astype('datetime64[D]'))
nt1 = 12000
model = waterNet.WaterNet1(nh)
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
    qP = model(pT, t1T, t2T, eT)
    loss = lossFun(qP, qT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)
    
    w = model.w
    gm = torch.exp(w[:nh])+1
    ge = torch.sigmoid(w[nh:nh*2])
    go = torch.sigmoid(w[nh*2:nh*3])
    ga = torch.softmax(w[nh*3:], dim=0)
    print(ga.max(), go[ga.argmax()])



model.eval()
qT = torch.from_numpy(Q[nt1:, None]).float().cuda()
pT = torch.from_numpy(P[nt1:, None]).float().cuda()
t1T = torch.from_numpy(T1[nt1:, None]).float().cuda()
t2T = torch.from_numpy(T2[nt1:, None]).float().cuda()
eT = torch.from_numpy(E[nt1:, None]).float().cuda()

qP = model(pT, t1T, t2T, eT)
model.zero_grad()
loss = lossFun(qP, qT)
yP = qP[:, 0].detach().cpu().numpy()


tt = df.index.values[nt1:]
fig, axes = plt.subplots(2, 1)
axes[0].plot(tt, P[nt1:])
axes[0].plot(tt, yP, '-r')
axes[1].plot(tt, Q[nt1:])
axes[1].plot(tt, yP, '-r')
fig.show()

w = model.w

gm = torch.exp(w[:nh])+1
ge = torch.sigmoid(w[nh:nh*2])
go = torch.sigmoid(w[nh*2:nh*3])
ga = torch.softmax(w[nh*3:], dim=0)

utils.stat.calNash(yP, Q[nt1:])
