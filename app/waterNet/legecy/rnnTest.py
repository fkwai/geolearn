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

p = (P-np.nanmin(P))/(np.nanmax(P)-np.nanmin(P))
q = (Q-np.nanmin(Q))/(np.nanmax(Q)-np.nanmin(Q))
e = (E-np.nanmin(E))/(np.nanmax(E)-np.nanmin(E))
# sn = 1e-5
# logp = np.log(P+sn)
# logq = np.log(Q+sn)
# p = (logp-np.nanmin(logp))/(np.nanmax(logp)-np.nanmin(logp))
# q = (logq-np.nanmin(logq))/(np.nanmax(logq)-np.nanmin(logq))

importlib.reload(waterNet)
nh = 3
nbatch = 1
rho = 30
nt = len(P)
model = waterNet.RnnModel(nh)
model = model.cuda()
optim = torch.optim.Adadelta(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=1e-1)

lossFun = crit.RmseLoss2D().cuda()

model.train()
for kk in range(100):
    iT = np.random.randint(0, nt-rho, [nbatch])
    xTemp = np.full([rho, nbatch], np.nan)
    yTemp = np.full([rho, nbatch], np.nan)
    for k in range(nbatch):
        xTemp[:, k] = p[iT[k]+1:iT[k]+rho+1]
        yTemp[:, k] = q[iT[k]+1:iT[k]+rho+1]
    xT = torch.from_numpy(xTemp).float().cuda()
    yT = torch.from_numpy(yTemp).float().cuda()
    model.zero_grad()
    yP, hP = model(xT)
    loss = lossFun(yP, yT)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(loss)


model.eval()
xT = torch.from_numpy(p[:, None]).float().cuda()
yT = torch.from_numpy(q[:, None]).float().cuda()
yOut, hOut = model(xT)
loss = lossFun(yOut, yT)
yP = yOut[:, 0].detach().cpu().numpy()


fig, axes = plt.subplots(2, 1)
axes[0].plot(yP, '-r')
axes[0].plot(p)
axes[1].plot(yP, '-r')
axes[1].plot(q)
fig.show()

dictPar = model.state_dict()
dictPar.keys()
w_i = dictPar['rnn.w_i'].detach().cpu().numpy()
w_h = dictPar['rnn.w_h'].detach().cpu().numpy()
w_o = dictPar['rnn.w_o'].detach().cpu().numpy()
b_i = dictPar['rnn.b_i'].detach().cpu().numpy()
b_h = dictPar['rnn.b_h'].detach().cpu().numpy()
b_o = dictPar['rnn.b_o'].detach().cpu().numpy()
# dictPar['rnn.b_h']
# dictPar['rnn.b_o']

x = np.ones([1, 1])
h = np.zeros([1, nh])
a = x*w_i+b_i
b = h*w_h+b_h
c = a+b
d = c*w_o+b_o

torch.sum(dictPar['rnn.w_h'])
