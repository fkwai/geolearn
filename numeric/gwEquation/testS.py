
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
Kg = np.ones([ny+2, nx+2])
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
for i in range(l2+1, nx-1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]
    b[j*nx+i] = h2*w01[j, i]
for i in range(l1, l2+1):
    A[j*nx+i, j*nx+i] = w11[j, i]
    A[j*nx+i, j*nx+i-1] = w10[j, i]
    A[j*nx+i, j*nx+i+1] = w12[j, i]
    A[j*nx+i, (j+1)*nx+i] = w21[j, i]+w01[j, i]
# left top
[j, i] = [0, 0]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i+1] = w12[j, i]+w10[j, i]
A[j*nx+i, (j+1)*nx+i] = w21[j, i]
b[j*nx+i] = h1*w01[j, i]
# right top
[j, i] = [0, nx-1]
A[j*nx+i, j*nx+i] = w11[j, i]
A[j*nx+i, j*nx+i-1] = w10[j, i]+w12[j, i]
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
cb1 = ax.imshow(A, vmin=-1, vmax=1)
fig.show()
