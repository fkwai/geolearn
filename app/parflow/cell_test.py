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
temp = data.wtd
temp[~maskS] = np.nan
fig, ax = plt.subplots(1, 1)
ind = np.where(temp == 0)
cb = ax.pcolor(temp)
ax.plot(ind[1], ind[0], 'r*')

fig.colorbar(cb)
fig.show()


outflow = data.overland_flow_grid()
outflow[~maskS] = np.nan
fig, ax = plt.subplots(1, 1)
ax.pcolor(outflow)
fig.show()


# balance for subsurface storage
# total storage
data.time = 0
storage = data.subsurface_storage
s0 = storage.sum()
f = 3.6
mat = np.zeros([nt, 6])
temp = 0
for k in range(nt):
    data.time = k + 1
    data.forcing_time = k
    storage = data.subsurface_storage
    outflow = data.overland_flow()
    evap = data.clm_output('qflx_tran_veg') + data.clm_output('qflx_evap_soi')
    # evap = data.clm_output('qflx_evap_soi')
    infi = data.clm_output('qflx_infl')
    s1 = storage.sum()
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 2] = (infi * maskS).sum() * dx * dy * f
    mat[k, 3] = outflow
    temp = temp + mat[k, 2] - mat[k, 1]
    # temp = temp - mat[k, 1]
    mat[k, 4] = temp
fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 0], label='storage')
ax.plot(mat[:, 4], label='storage calculated')
ax.legend()
fig.show()

temp = mat[:, 2]
temp[temp < 0] = 0
fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 0] + np.cumsum(mat[:, 1]), label='storage - evap')
ax.plot(np.cumsum(temp), label='infi')
ax.legend()
fig.show()

fig, axes = plt.subplots(2, 1)
dataPlot = np.concatenate([mat[:, 1:3]], axis=-1)
# [mat[:, 1], mat[:, 2], mat[:, 3], mat[:, 5]]
labelLst = ['evap', 'infi']
axplot.multiTS(axes, tAry, dataPlot, labelLst=labelLst)
fig.show()
