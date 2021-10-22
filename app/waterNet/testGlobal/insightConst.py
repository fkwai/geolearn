
w = model.fc(xcP)
gm = torch.exp(w[:, :nh])+1
ge = torch.sigmoid(w[:, nh:nh*2])*2
go = torch.sigmoid(w[:, nh*2:nh*3])
gl = torch.exp(w[:, nh*3:nh*4])*2
ga = torch.softmax(w[:, nh*4:nh*5], dim=1)
gb = torch.sigmoid(w[:, nh*5:nh*6])
qb = torch.relu(w[:, -1])/nh
kb = torch.sigmoid(w[:, nh*6:nh*7])/10

a = ga.detach().cpu().numpy()
l = gl.detach().cpu().numpy()
b = kb.detach().cpu().numpy()
m = gm.detach().cpu().numpy()

x = go.detach().cpu().numpy()
d = np.sum(x*a, axis=1)
for var in DF.varG:
    y = DF.g[:, DF.varG.index(var)]
    print(np.corrcoef(d, y)[0, 1], var)

y = DF.g[:, DF.varG.index('PLANTNLCD06')]
fig, ax = plt.subplots(1, 1)
ax.plot(d, y, '*')
fig.show()
np.corrcoef(d, y)

k = 0
x0 = torch.ones(ng).cuda()*0.5
x1 = torch.zeros(ng).cuda()
x1[k] = 1
w0 = model.fc(x0)
w1 = model.fc(x1)
gl0 = torch.exp(w0[nh*3:nh*4])*2
gl1 = torch.exp(w1[nh*3:nh*4])*2
torch.sum(w1-w0)
