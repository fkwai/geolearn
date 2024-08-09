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
run_name = "14080103"
sd = "2005-10-01"
ed = "2006-10-01"
tAry = pd.date_range(start=sd, end=ed, freq='H')[:-1]

work_dir=r'/home/kuai/GitHUB/EcoSLIM/Examples/ParFlow_MixedBCs/mixedBCs_hillslope2D_transient'
work_dir = os.path.join(kPath.dirParflow, run_name, 'outputs')
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))
data = run.data_accessor
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape
xx = np.arange(0, nx + 1) * dx
zz = np.insert(np.cumsum(dz), 0, 0)
xm, zm = np.meshgrid(xx, zz)
slopeX = data.slope_x
maskS = data.mask[-1, :, :]
maskS[maskS > 0] = 1
maskS = maskS.astype(bool)

data.time = 1440
pressure = data.pressure
saturation = data.saturation
velx = data._pfb_to_array(f'{data._name}.out.velx.{data._ts}.pfb')
vely = data._pfb_to_array(f'{data._name}.out.vely.{data._ts}.pfb')
velz = data._pfb_to_array(f'{data._name}.out.velz.{data._ts}.pfb')
qi = dx * dy * velz[-1, :, :]

fig, ax = plt.subplots(1, 1)
temp = velz[1, :, :].copy()
# temp[~maskS] = np.nan
cb = ax.pcolor(temp)
fig.colorbar(cb)
fig.show()

fig, ax = plt.subplots(1, 1)
temp = saturation[-1, :, :].copy()
temp[~maskS] = np.nan
cb = ax.pcolor(temp)
fig.colorbar(cb)
fig.show()

data.time = 1440

nt = 1440
mat = np.zeros([nt, 11])
for k in range(nt):
    data.time = k + 1
    velz = data._pfb_to_array(f'{data._name}.out.velz.{data._ts}.pfb')
    for kk in range(11):
        temp = velz[kk, :, :]
        temp[~maskS] = np.nan        
        mat[k, kk] = np.nansum(temp)
cmap = plt.cm.jet
cLst = cmap(np.linspace(0, 1, 11))
fig, ax = plt.subplots(1, 1)
for kk in range(11):
    ax.plot(mat[:, kk], label=kk, color=cLst[kk])
ax.legend()
fig.show()

edge_south = np.maximum(0, np.diff(maskS, axis=0, prepend=0))
edge_north = np.maximum(0, -np.diff(maskS, axis=0, append=0))
edge_west = np.maximum(0, np.diff(maskS, axis=1, prepend=0))
edge_east = np.maximum(0, -np.diff(maskS, axis=1, append=0))
qnorth = vely * dx * dz[:, np.newaxis, np.newaxis]

np.roll(qnorth, -1, axis=0).shape
qsouth[0, :-1, :][edge_north == 1]
flux_north = np.sum(np.maximum(0, np.roll(qnorth, -1, axis=0)[np.where(edge_north == 1)]))
flux_south = np.sum(np.maximum(0, -qnorth[np.where(edge_south == 1)]))
flux_west = np.sum(np.maximum(0, -qeast[np.where(edge_west == 1)]))
flux_east = np.sum(np.maximum(0, np.roll(qeast, -1, axis=1)[np.where(edge_east == 1)]))

fig, ax = plt.subplots(1, 1)
cb = ax.pcolor(edge_north)
fig.colorbar(cb)
fig.show()

# balance for subsurface storage
nt = 8760
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
    velz = data._pfb_to_array(f'{data._name}.out.velz.{data._ts}.pfb')
    infi = dx * dy * velz[-1, :, :]
    evap = data.clm_output('qflx_tran_veg')
    s1 = storage.sum()
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 2] = infi.sum()
    mat[k, 3] = outflow
    temp = temp + mat[k, 2] - mat[k, 1]
    # temp = temp  - mat[k, 1]
    mat[k, 4] = temp
fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 0], label='storage')
ax.plot(mat[:, 4], label='storage calculated')
ax.legend()
fig.show()


fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 2], label='storage')

ax.legend()
fig.show()
