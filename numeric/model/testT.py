
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

# create K
Kg = (np.random.rand(ny+2, nx+2)+0.1)*10
# Kg = np.ones([ny+2, nx+2])
K = Kg[1:-1, 1:-1]
Kx = (Kg[2:, 1:-1]-Kg[:-2, 1:-1])/2/d
Ky = (Kg[1:-1, 2:]-Kg[1:-1, :-2])/2/d


w11 = -4*K/d**2
w10 = -Kx/2/d+K/d**2
w12 = Kx/2/d+K/d**2
w01 = -Ky/2/d+K/d**2
w21 = Ky/2/d+K/d**2

A = np.zeros([ny*nx, ny*nx])
b = np.zeros([ny*nx])

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
    A[j*nx+i, j*nx+i+1] = w12[j, i]+w10[j, i]
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
# right Neumann
i = nx-1
for j in range(1, ny-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]+w12[j, i]
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
# bottom Neumann
j = ny-1
for i in range(1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j-1)*nx+i] = w01[j, i]+w21[j, i]
# top Dirichlet + Neumann
j = 0
for i in range(1, l1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
    b[j*nx+i] = h1*w01[j, i]
for i in range(l1, l2+1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i] + w01[j, i]
for i in range(l2+1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
    b[j*nx+i] = h2*w01[j, i]
# left top
[j, i] = [0, 0]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i+1] = w12[j, i] + w10[j, i]
A[j*nx+i, (j+1)*nx+i] = w21[j, i]
b[j*nx+i] = h1*w01[j, i]
# right top
[j, i] = [0, nx-1]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i-1] = w10[j, i] + w12[j, i]
A[j*nx+i, (j+1)*nx+i] = w21[j, i]
b[j*nx+i] = h2*w01[j, i]
# left bottom
[j, i] = [ny-1, 0]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i+1] = w12[j, i]+w10[j, i]
A[j*nx+i, (j-1)*nx+i] = w01[j, i]+w21[j, i]
# right bottom
[j, i] = [ny-1, nx-1]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i-1] = w10[j, i]+w12[j, i]
A[j*nx+i, (j-1)*nx+i] = w01[j, i]+w21[j, i]

# static state
V = np.linalg.solve(A, -b)
H = V.reshape([ny, nx])

# plot
fig, ax = plt.subplots(1, 1)
cb1 = ax.imshow(H, vmin=0, vmax=20)
ax.contour(H, colors='r')
ax.plot([l1, l2], [0-d/2, 0-d/2], 'k-', linewidth=5)
# ax.set_ylim([20-d/2, -1])
fig.colorbar(cb1, ax=ax)
fig.show()
fig, ax = plt.subplots(1, 1)
cb1 = ax.imshow(np.linalg.inv(A))
fig.show()

#####################
# transient
HLst = list()
HLst.append(H)
dt = 1/24
nt = 24*10
t = np.arange(nt)*dt
h1t = h1+np.sin(2*np.pi*t)*h1
Ss = (np.random.rand(ny, nx)+1)*0.1
Sv = Ss.reshape([ny*nx])
# Sv = 0.1

j = 0
for i in range(0, l1):
    b[j*nx+i] = 10*w01[j, i]

# change boundary condition
for k in range(nt):
    # forward
    # A1 = A*dt/Sv+np.eye(nx*ny)
    # b1 = b*dt/Sv
    # backward
    A2 = (A.T*dt/Sv).T-np.eye(nx*ny)
    A1 = -np.linalg.inv(A2)
    b1 = np.matmul(A1, b*dt/Sv)
    # CN
    # A1 = (A.T*dt/Sv).T-np.eye(nx*ny)
    # B1 = (A.T*dt/Sv).T+np.eye(nx*ny)
    # A2 = np.matmul(A1, np.linalg.inv(B1))

    V2 = np.matmul(A1, V)+b1
    H2 = V2.reshape([ny, nx])
    HLst.append(H2)
    V = V2

fig, ax = plt.subplots(1, 1)
cb1 = ax.imshow(A2, vmin=0, vmax=20)
fig.show()


cz = np.arange(0, 20, 2)
cz2 = np.arange(0, 20, 4)
fig, axes = plt.subplots(3, 1)
cb1 = axes[0].imshow(HLst[0], vmin=0, vmax=20)
cs1 = axes[0].contour(HLst[0],  cz, colors='r')
axes[0].clabel(cs1, cz2, inline=True, fmt='%d', fontsize=10)
fig.colorbar(cb1, ax=axes[0])
k = 50
cb2 = axes[1].imshow(HLst[k], vmin=0, vmax=20)
cs2 = axes[1].contour(HLst[k], cz, colors='r')
axes[1].clabel(cs2, cz2, inline=True, fmt='%d', fontsize=10)
fig.colorbar(cb2, ax=axes[1])
cb3 = axes[2].imshow(HLst[-1], vmin=0, vmax=20)
cs3 = axes[2].contour(HLst[-1], cz, colors='r')
axes[2].clabel(cs3, cz2, inline=True, fmt='%d', fontsize=10)
fig.colorbar(cb3, ax=axes[2])
fig.show()


# def animate(k):
#     plt.clf()
#     day = int(t[k])
#     hr = round((t[k]-day)*24)
#     plt.title('head at time {:.0f}d {:.0f}h'.format(day, hr))
#     plt.imshow(HLst[k], vmin=0, vmax=20)
#     cs = plt.contour(HLst[k], cz, colors='r')
#     plt.clabel(cs, cz2, inline=True, fmt='%d', fontsize=10)


# anim = animation.FuncAnimation(
#     plt.figure(), animate, interval=1, frames=nt, repeat=False)
# anim.save('temp.gif')
