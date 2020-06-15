"""
x1-x2**3=0
x1**2+x2**2-1=0
"""
import numpy as np
from numeric import es


def F(x):
    temp = [x[0]-x[1]**3, x[0]**2+x[1]**2-1]
    return np.array(temp)


def DF(x):
    temp = [
        [1, -3*x[1]**2],
        [2*x[0], 2*x[1]]
    ]
    return np.array(temp)


x = [2, 1]
for k in range(10):
    s = es.gaussElimination(DF(x), -F(x))
    x = x+s
    print(x)

x = es.newton(F, DF, [1,2], n=10)