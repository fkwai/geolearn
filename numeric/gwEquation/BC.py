import matplotlib.animation as animation
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.utils import draw
import matplotlib

nx = 100
ny = 20
d = 10

NX=50
NZ=20
dl = draw.LineRegDrawer(bbox=[0, 1, 0, 1], nPoint=NX)

# unconfined
dl = draw.LineRegDrawer(bbox=[0, 1, 0, 1], nPoint=nx + 2)
xd, yd = dl.draw_line(order=2)
# HA=np.random.rand(nx + 2) +10
HA=yd*20
KA=np.random.normal(loc=0.05,scale=0.0001,size=nx+2)*500
Sy=np.random.normal(loc=0.25,scale=0.0001,size=nx)
# Boundary condition
HA[0] = HA[1]
HA[-1] = 5
K=KA[1:-1]

nt=30
cmap=matplotlib.cm.get_cmap('jet')
hLst,qLst=[],[]

fig, axes = plt.subplots(2, 1)
axes[0].plot(HA[1:-1],color=cmap(0))
for k in range(nt):  
    nts=100
    HA=HA+np.random.normal(loc=1,scale=1,size=nx+2)*0.1
    for i in range(nts):
        H=HA[1:-1]
        HA[0] = HA[1]
        HA[-1] = 5  
        dhdx=(HA[2:]-HA[:-2])/2/d
        dhdx2=(HA[2:]-2*HA[1:-1]+HA[:-2])/d**2
        dkdx=(KA[2:]-KA[:-2])/2/d
        dhdt=K*H*dhdx2 + H*dkdx*dhdx + K*dhdx**2
        HA[1:-1]=HA[1:-1]+dhdt/Sy/nts                 
        Q=-K*H*dhdx
    H=HA[1:-1]
    axes[0].plot(H,color=cmap(k/nt))
    print(H)
    axes[1].plot(Q,color=cmap(k/nt))
    hLst.append(H.copy())
    qLst.append(Q)
fig.show()



hh=np.array(hLst)
qq=np.array(qLst)

fig,ax=plt.subplots(1,1)
ax.imshow(hh)
fig.show()


fig,ax=plt.subplots(1,1)
for k in range(10,nx+1,5):
    x=np.mean(hh[:,:k],axis=1)
    y=qq[:,k]
    ax.plot(y,x,'*-',label=str(k))
x=np.mean(hh,axis=1)
y=qq[:,-1]
ax.plot(y,x,'*-',label='end')
ax.legend()
fig.show()


# fig, ax = plt.subplots(1, 1)
# ax.plot(ss)
# fig.show()
# fig, ax = plt.subplots(1, 1)
# lns=np.log(-np.log(ss))
# ax.plot(lns)
# ax.plot([0,nt],[lns[0],lns[-1]])
# fig.show()

fig,ax=plt.subplots(1,1)
ax.plot(qLst,sLst,'*')
fig.show()


fig,ax=plt.subplots(1,1)
ax.plot(q1Lst,s1Lst,'*')
fig.show()