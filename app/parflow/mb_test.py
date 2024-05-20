import sys
import os
import parflow
from parflow import Run
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from hydroDL import kPath
import parflow.tools.hydrology as pfhydro

nt = 120

# mesh
run_name = '15060202'
run_name = "10180001"

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
mat = np.zeros([nt, 5])
temp = 0
for k in range(nt):
    data.time = k + 1
    data.forcing_time = k
    storage = data.subsurface_storage
    pond = data.surface_storage
    outflow = data.overland_flow()
    s1 = storage.sum() + pond.sum()
    # s1 = storage.sum()
    evap = data.clm_output('qflx_evap_tot')
    p = data.clm_forcing('APCP')
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 2] = outflow * f
    mat[k, 3] = (p * maskS).sum() * dx * dy * f
    temp = temp - mat[k, 2] - mat[k, 1] + mat[k, 3]
    mat[k, 4] = temp
fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 1], label='evap')
ax.plot(mat[:, 2], label='outflow')
ax.plot(mat[:, 3], label='prcp')
ax.plot(mat[:, 0], label='storage')
ax.plot(mat[:, 4], label='storage calculated')
ax.legend()
fig.show()

# overland flow
slopeX = data.slope_x
slopeY = data.slope_y

data.elevation
data.t = 100
outflow = data.overland_flow_grid()
fig,ax=plt.subplots(1,1)
ax.pcolor(outflow)
fig.show()