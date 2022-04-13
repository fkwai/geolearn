import numpy as np

n = 9
out = 0
for k in range(1, n+1):
    out = out+np.math.factorial(n)/np.math.factorial(np.max([n-k, k]))
out

n = 14
k = 3
np.math.factorial(n)/np.math.factorial(np.max([n-k, k]))

a = np.random.randn(400, 3)
b = np.random.randn(400, 2)
np.divide(a, b)

bi = np.linalg.pinv(b)

np.matmul(bi, b)

c = np.matmul(bi, a)
