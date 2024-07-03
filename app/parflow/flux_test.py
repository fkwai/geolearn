import sys
import os
import parflow
from parflow import Run
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from hydroDL import kPath
import parflow.tools.hydrology as pfhydro
import pandas as pd
from hydroDL.post import axplot

# from yaml file
N = 3


nt = 8760

# mesh
# run_name = '15060202'
run_name = "10180001"
sd = "2005-10-01"
ed = "2006-10-01"
tAry = pd.date_range(start=sd, end=ed, freq='H')[:-1]


work_dir = os.path.join(kPath.dirParflow, run_name, 'outputs')
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))
data = run.data_accessor
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape
xx = np.arange(0, nx + 1) * dx
zz = np.insert(np.cumsum(dz), 0, 0)
xm, zm = np.meshgrid(xx, zz)
slopeX = data.slope_x
maskS = data.mask[0, :, :]
maskS[maskS > 0] = 1
maskS = maskS.astype(bool)

data.time = 1440
pressure = data.pressure
saturation = data.saturation

sres = data._pfb_to_array(f'{data._name}.out.sres.pfb')
ssat = data._pfb_to_array(f'{data._name}.out.ssat.pfb')
Nvg = 3
Alpha = 1
m = 1.0 - 1.0 / Nvg

opahn = 1 + (Alpha * np.abs(pressure)) ** Nvg
ahnm1 = (Alpha * np.abs(pressure)) ** (Nvg - 1)
krel = (1 - ahnm1 / (opahn) ** m) ** 2 / opahn ** (m / 2)


fig, ax = plt.subplots(1, 1)
temp = opahn
temp[data.mask == 0] = np.nan
cb = ax.pcolor(temp[0, :, :])
fig.colorbar(cb)
fig.show()
