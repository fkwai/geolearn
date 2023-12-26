import numpy as np

np.random.seed(1000)  # For debugging and reproducibility

N = 1000

Z = np.array([0])
for n in range(N):
    prev_z = Z[len(Z) - 1]
    if prev_z == 0:
        post_z = np.random.choice(3, size=1, p=[0.7, 0.15, 0.15])
    elif prev_z == 1:
        post_z = np.random.choice(3, size=1, p=[0.0, 0.5, 0.5])
    elif prev_z == 2:
        post_z = np.random.choice(3, size=1, p=[0.3, 0.35, 0.35])
    Z = np.append(Z, post_z)
Z

X = np.empty((0,2))
for z_n in Z:
    if z_n == 0:
        x_n = np.random.multivariate_normal(
            mean=[16.0, 1.0],
            cov=[[4.0,3.5],[3.5,4.0]],
            size=1)
    elif z_n == 1:
        x_n = np.random.multivariate_normal(
            mean=[1.0, 16.0],
            cov=[[4.0,0.0],[0.0,1.0]],
            size=1)
    elif z_n ==2:
        x_n = np.random.multivariate_normal(
            mean=[-5.0, -5.0],
            cov=[[1.0,0.0],[0.0,4.0]],
            size=1)
    X = np.vstack((X, x_n))
X