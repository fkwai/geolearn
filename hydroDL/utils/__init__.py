from . import time
from . import stat
from .array import *
import numpy as np


def index2d(ind, ny, nx):
    iy = np.floor(ind / nx)
    ix = np.floor(ind % nx)
    return int(iy), int(ix)


def sameClass(obj1, obj2):
    # sth wrong with type or isinstance
    # used for moduled objects
    if (obj1.__name__ == obj2.__name__) and (obj1.__module__ == obj2.__module__):
        return True

# class TimedOutExc(Exception):
#     pass

# def deadline(timeout, *args):
#     def decorate(f):
#         def handler(signum, frame):
#             raise TimedOutExc()

#         def new_f(*args):
#             signal.signal(signal.SIGALRM, handler)
#             signal.alarm(timeout)
#             return f(*args)
#             signal.alarm(0)

#         new_f.__name__ = f.__name__
#         return new_f
#     return decorate
