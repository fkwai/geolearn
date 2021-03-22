
import matplotlib.animation as animation
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt

nx = 40
ny = 20
d = 1
l1 = 20
l2 = 30
h1 = 20
h2 = 5
K = (np.random.rand(ny, nx)+.5)*5
# K = np.ones([ny, nx])

# create K
Kg = np.ones([ny+2, nx+2])
Kg[1:-1, 1:-1] = K
Kx = (Kg[2:, 1:-1]-Kg[:-2, 1:-1])/2/d
Ky = (Kg[1:-1, 2:]-Kg[1:-1, :-2])/2/d

w11 = -4*K/d**2
w10 = -Kx/2/d+K/d**2
w12 = Kx/2/d+K/d**2
w01 = -Ky/2/d+K/d**2
w21 = Ky/2/d+K/d**2

A = np.zeros([ny*nx, ny*nx])
R = np.zeros([ny*nx])

# dumb way
# center
for j in range(1, ny-1):
    for i in range(1, nx-1):
        A[j*nx+i, j*nx+i] = w11[j, i]
        A[j*nx+i, j*nx+i-1] = w10[j, i]
        A[j*nx+i, j*nx+i+1] = w12[j, i]
        A[j*nx+i, (j-1)*nx+i] = w01[j, i]
        A[j*nx+i, (j+1)*nx+i] = w21[j, i]
# left Neumann
i = 0
for j in range(1, ny-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]*2
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
# right Neumann
i = nx-1
for j in range(1, ny-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]*2
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
# bottom Dirichlet
j = ny-1
for i in range(1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]
    R[j*nx+i] = -h2
# top Dirichlet + Neumann
j = 0
for i in range(1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
    R[j*nx+i] = -h1
for i in range(l2+1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
    R[j*nx+i] = -h2
for i in range(l1, l2+1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]*2
# left top
[j, i] = [0, 0]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i+1] = w12[j, i]*2
A[j*nx+i, (j+1)*nx+i] = w21[j, i]
R[j*nx+i] = -h1
# right top
[j, i] = [0, nx-1]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i-1] = w10[j, i]*2
A[j*nx+i, (j+1)*nx+i] = w21[j, i]
R[j*nx+i] = -h2
# left bottom
[j, i] = [ny-1, 0]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i+1] = w12[j, i]*2
A[j*nx+i, (j-1)*nx+i] = w01[j, i]*2
# right bottom
[j, i] = [ny-1, nx-1]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i-1] = w10[j, i]*2
A[j*nx+i, (j-1)*nx+i] = w01[j, i]*2

V = np.linalg.solve(A, R)
H = V.reshape([ny, nx])

HLst = list()
HLst.append(H)
dt = 1/24
nt = 100
t = np.arange(nt)*dt
h1t = h1+np.sin(2*np.pi*t)*h1
Ss = np.random.rand(ny, nx)*0.01

Hg = np.zeros([ny+2, nx+2])
Hg[0, 1:l1] = h1
Hg[0, l2+1:] = h2
for k in range(nt):
    Hg = np.zeros([ny+2, nx+2])
    Hg[1:-1, 1:-1] = H
    Hg[1:-1, 0] = H[:, 0]
    Hg[1:-1, -1] = H[:, -1]
    Hg[0, 1:-1] = H[0, :]
    Hg[-1, 1:-1] = H[-1, :]
    Hg[0, 1:l1] = h1t[k]

    Hx = (Hg[1:-1, 2:]-Hg[1:-1, :-2])/2/d
    Hxx = (Hg[1:-1, 2:]+Hg[1:-1, :-2]-Hg[1:-1, 1:-1]*2)/d**2
    Hy = (Hg[2:, 1:-1]-Hg[:-2, 1:-1])/2/d
    Hyy = (Hg[2:, 1:-1]+Hg[:-2, 1:-1]-Hg[1:-1, 1:-1]*2)/d**2
    Ht = (Kx*Hx+K*Hxx+Ky*Hy+K*Hyy)*dt*Ss+H
    HLst.append(Ht)
    H = Ht


fig, axes = plt.subplots(3, 1)
cb1 = axes[0].imshow(HLst[0])
axes[0].contour(HLst[0], colors='r')
fig.colorbar(cb1, ax=axes[0])
cb2 = axes[1].imshow(HLst[50])
axes[1].contour(HLst[50], colors='r')
fig.colorbar(cb2, ax=axes[1])
cb3 = axes[2].imshow(HLst[-1])
axes[2].contour(HLst[-1], colors='r')
fig.colorbar(cb3, ax=axes[2])
fig.show()


def animate(k):
    plt.clf()
    plt.title('head at step {}'.format(k))
    plt.imshow(HLst[k], vmin=0, vmax=10)
    plt.contour(HLst[k], colors='r')


anim = animation.FuncAnimation(
    plt.figure(), animate, interval=1, frames=nt, repeat=False)
anim.save('temp.gif')


fig, axes = plt.subplots(3, 1)
cb1 = axes[0].imshow(HLst[0])
axes[0].contour(HLst[0], colors='r')
fig.colorbar(cb1, ax=axes[0])
cb2 = axes[1].imshow(HLst[50])
axes[1].contour(HLst[50], colors='r')
fig.colorbar(cb2, ax=axes[1])
cb3 = axes[2].imshow(HLst[-1])
axes[2].contour(HLst[-1], colors='r')
fig.colorbar(cb3, ax=axes[2])
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(t,h1t)
fig.show()