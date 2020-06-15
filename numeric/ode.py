import numpy as np


def forward1(tt, yy, dt, dF):
    dd = dt*dF(tt, yy)
    return yy+dd


def forward2(tt, yy, dt, dF):
    return yy+dt/2*(dF(tt, yy) + dF(tt+dt, yy+dt*dF(tt, yy)))
