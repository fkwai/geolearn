
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
K = (np.random.rand(ny, nx)+.5)*10
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


fig, axes = plt.subplots(2, 1)
cb1 = axes[0].imshow(K)
fig.colorbar(cb1, ax=axes[0])
cb2 = axes[1].imshow(H)
axes[1].contour(H, colors='r')
axes[1].plot([l1, l2], [0-d/2, 0-d/2], 'k-', linewidth=5)
fig.colorbar(cb2, ax=axes[1])
fig.show()

fig, ax = plt.subplots(1, 1)
AI = np.linalg.inv(A)
ax.imshow(AI)
fig.show()
