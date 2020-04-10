import numpy as np


def gaussElimination(A, b):
    A = A.copy()
    b = b.copy()
    n = A.shape[0]
    for j in range(0, n-1):
        for i in range(j+1, n):
            k = A[i, j]/A[j, j]
            A[i, :] = A[i, :]-A[j, :]*k
            b[i] = b[i]-b[j]*k
    x = np.zeros([n])
    for j in range(n-1, -1, -1):
        for k in range(j+1, n):
            b[j] = b[j]-x[k]*A[j, k]
        x[j] = b[j]/A[j, j]
    return x


def LU(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for j in range(0, n-1):
        for i in range(j+1, n):
            L[i, j] = U[i, j]/U[j, j]
            U[i, :] = U[i, :]-U[j, :]*L[i, j]
    return L, U


def jacobi(A, b, x, n=10):
    Di = np.diag(1/np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    for k in range(n):
        x = Di.dot(b-(L+U).dot(x))
    return x
