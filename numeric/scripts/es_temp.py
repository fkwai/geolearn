import numpy as np
from numeric import es
import importlib
A = np.array([[3, 1, -1],
              [2, 4, 1],
              [-1, 2, 5]])
b = [4, 1, 1]
x = np.array([2, -1, 1])

B = np.linalg.inv(A).dot(b)

importlib.reload(es)

x = es.gaussElimination(A, b)
L, U = es.LU(A)
x = es.jacobi(A, b, [0, 0, 0], n=20)
