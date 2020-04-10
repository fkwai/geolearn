
import numpy as np


def test(a):
    a[0] = a[0]+1
    b = a
    return b


def test2(a):
    a = a+1
    b = a
    return b


a = np.array([1, 2])
# a = np.array([1])
# a=[1,2,3]
b = test2(a)


a = (1, 2)
b = list(a)
c = b
id(a)
id(b)
id(c)

a = np.array([1, 2])
b = a.copy()
id(a)
id(b)
a = a+1
id(a)
