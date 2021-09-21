import numpy as np
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
import torch
import importlib
from hydroDL.model import waterNet, crit

# test case
siteNo = '07241550'
df = dbBasin.readSiteTS(siteNo, ['pr', 'runoff', 'pet'])
P = df['pr'].values
Q = df['runoff'].values
E = df['pet'].values


p = (P-np.nanmin(P))/(np.nanpercentile(P, 90)-np.nanmin(P))
q = (Q-np.nanmin(Q))/(np.nanpercentile(Q, 90)-np.nanmin(Q))
e = (E-np.nanmin(E))/(np.nanpercentile(E, 90)-np.nanmin(E))
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
nh = 8
nbatch = 50
rho = 100
nt = len(P)
nt1 = 12000
model = waterNet.RnnModel(nh)
model = model.cuda()
optim = torch.optim.Adagrad(model.parameters(), lr=0.1)

lossFun = torch.nn.MSELoss().cuda()

model.train()
for kk in range(1000):
    iT = np.random.randint(0, nt1-rho, [nbatch])
    xTemp = np.full([rho, nbatch], np.nan)
    yTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        xTemp[:, k] = p[iT[k]+1:iT[k]+rho+1]
        yTemp[:, k] = q[iT[k]+1:iT[k]+rho+1]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    model.zero_grad()
    # model.state_dict()['rnn.w_i']
    # model.state_dict()['rnn.w_o']
    yP, hP = model(xT)
    loss = lossFun(yP, yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)


model.eval()
xT = torch.from_numpy(p[nt1:, None]).float().cuda()
yT = torch.from_numpy(q[nt1:, None]).float().cuda()
yOut, hOut = model(xT)
model.zero_grad()
loss = lossFun(yOut, yT)
yP = yOut[:, 0].detach().cpu().numpy()


fig, axes = plt.subplots(2, 1)
axes[0].plot(yP, '-r')
axes[0].plot(p[nt1:])
axes[1].plot(yP, '-r')
axes[1].plot(q[nt1:])
fig.show()

dictPar = model.state_dict()
dictPar.keys()
w_i = dictPar['rnn.w_i'].detach().cpu().numpy().flatten()
w_o = dictPar['rnn.w_o'].detach().cpu().numpy().flatten()
b_i = dictPar['rnn.b_i'].detach().cpu().numpy().flatten()
b_o = dictPar['rnn.b_o'].detach().cpu().numpy().flatten()
# dictPar['rnn.b_h']
# dictPar['rnn.b_o']

x = np.ones([1, 1])
h = np.zeros([1, nh])
