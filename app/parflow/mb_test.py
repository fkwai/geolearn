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
t = pd.date_range(start=sd, end=ed, freq='H')[:-1]


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

data.time = 120
temp = data.wtd
temp[~maskS] = np.nan
fig, ax = plt.subplots(1, 1)
cb = ax.pcolor(temp)
fig.colorbar(cb)
fig.show()

outflow = data.overland_flow_grid()
outflow[~maskS] = np.nan
fig, ax = plt.subplots(1, 1)
ax.pcolor(outflow)
fig.show()


# check the total precipitation
input_dir = os.path.join(kPath.dirParflow, run_name, 'inputs', 'forcing')
# find all CW3E.APCP.00000x_to_0000xx.pfb and extract 00000x
fileList = os.listdir(input_dir)
fileList = [x for x in fileList if 'Press' in x]
fileList = [x for x in fileList if x.endswith('.pfb')]
tLst, pLst = [], []
for file in fileList:
    tempStr = file.split('.')
    t = int(tempStr[2][:6])
    tLst.append(t)
    p = parflow.read_pfb(os.path.join(input_dir, file)).sum()
    pLst.append(p)
out = np.array([tLst, pLst]).T
out.sort(axis=0)
fig, ax = plt.subplots(1, 1)
ax.plot(out[:, 0] / 24, out[:, 1])
fig.show()

# file = '/home/kuai/work/parflow/LW/input/NLDAS/NLDAS.APCP.000097_to_000120.pfb'
file = r'/home/kuai/work/parflow/15060202/inputs/forcing/CW3E.APCP.000001_to_000024.pfb'
p = parflow.read_pfb(file)
p.sum()

# total storage
data.time = 0
storage = data.subsurface_storage
pond = data.surface_storage
s0 = storage.sum() + pond.sum()
f = 3.6
mat = np.zeros([nt, 6])
temp = 0
for k in range(nt):
    data.time = k + 1
    data.forcing_time = k
    storage = data.subsurface_storage
    pond = data.surface_storage
    outflow = data.overland_flow()
    # s1 = storage.sum()
    evap = data.clm_output('qflx_evap_tot')
    p = data.clm_forcing('APCP')
    swe = data.clm_output('swe_out')
    s1 = storage.sum() + pond.sum()
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 5] = (swe * maskS).sum() * dx * dy / 1000
    mat[k, 2] = outflow
    mat[k, 3] = (p * maskS).sum() * dx * dy * f
    temp = temp - mat[k, 2] - mat[k, 1] + mat[k, 3]
    mat[k, 4] = temp
fig, ax = plt.subplots(1, 1)
# axes[0].plot(t,mat[:, 1], label='evap')
# axes[1].plot(t,mat[:, 2], label='outflow')
# axes[2].plot(t,mat[:, 3], label='prcp')
# axes[3].plot(t,mat[:, 5], label='swe')
ax.plot(t,mat[:, 0], label='storage')
ax.plot(t,mat[:, 4] - mat[:, 5], label='storage calculated')
ax.legend()
fig.show()

fig, axes = plt.subplots(4, 1)
dataPlot=np.concatenate([mat[:, 1:4], mat[:, 5:6]], axis=-1)
# [mat[:, 1], mat[:, 2], mat[:, 3], mat[:, 5]]
labelLst=['evap', 'outflow', 'prcp', 'swe']
axplot.multiTS(axes, t, dataPlot, labelLst=labelLst)
fig.show()


storage = data.subsurface_storage
pond = data.surface_storage
mS0 = storage.sum(axis=0) + pond
mE,mQ,mP,mS = [np.zeros([ny,nx]) for x in range(4)]
for k in range(nt):
    data.time = k + 1
    data.forcing_time = k
    storage = data.subsurface_storage
    pond = data.surface_storage
    outflow = data.overland_flow()
    # s1 = storage.sum()
    evap = data.clm_output('qflx_evap_tot')
    p = data.clm_forcing('APCP')
    swe = data.clm_output('swe_out')
    s1 = storage.sum() + pond.sum()
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 5] = (swe * maskS).sum() * dx * dy / 1000
    mat[k, 2] = outflow
    mat[k, 3] = (p * maskS).sum() * dx * dy * f
    temp = temp - mat[k, 2] - mat[k, 1] + mat[k, 3]
    mat[k, 4] = temp
fig, ax = plt.subplots(1, 1)
# axes[0].plot(t,mat[:, 1], label='evap')
# axes[1].plot(t,mat[:, 2], label='outflow')
# axes[2].plot(t,mat[:, 3], label='prcp')
# axes[3].plot(t,mat[:, 5], label='swe')
ax.plot(t,mat[:, 0], label='storage')
ax.plot(t,mat[:, 4] - mat[:, 5], label='storage calculated')
ax.legend()
fig.show()

a= mat[1:, 0]-mat[:-1, 0]-mat[:-1, 3]+mat[:-1, 1]
b=mat[1:, 0]
fig, ax = plt.subplots(1, 1)
ax.plot(a,b)
fig.show()


# overland flow
slopeX = data.slope_x
slopeY = data.slope_y

data.elevation
data.t = 0
outflow = data.overland_flow_grid()
# outflow[~maskS] = np.nan

fig, ax = plt.subplots(1, 1)
ax.pcolor(outflow)
fig.show()
