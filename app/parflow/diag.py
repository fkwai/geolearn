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
from hydroDL import utils

run_name = "10180001"
sd = "2005-10-01"
ed = "2006-10-01"
tAry = pd.date_range(start=sd, end=ed, freq='H')[:-1]

work_dir = os.path.join(kPath.dirParflow, run_name, 'outputs')
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))
data = run.data_accessor

permX = data.computed_permeability_x
permY = data.computed_permeability_y
permZ = data.computed_permeability_z
slopeX = data.slope_x
slopeY = data.slope_y
porosity = data.computed_porosity
ss = data.specific_storage
mask = data.mask
manning = data.mannings
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape


# for a time step
data.time = 1000
pressure = data.pressure
saturation = data.saturation
velx = data._pfb_to_array(f'{data._name}.out.velx.{data._ts}.pfb')
vely = data._pfb_to_array(f'{data._name}.out.vely.{data._ts}.pfb')
velz = data._pfb_to_array(f'{data._name}.out.velz.{data._ts}.pfb')

# unsaturate
Nvg = 3
Alpha = 1
m = 1.0 - 1.0 / Nvg
opahn = 1 + (Alpha * np.abs(pressure)) ** Nvg
ahnm1 = (Alpha * np.abs(pressure)) ** (Nvg - 1)
krel = (1 - ahnm1 / (opahn) ** m) ** 2 / opahn ** (m / 2)
krel[saturation==1]=1
# flow c
pressMask = pressure.copy()
pressMask[data.mask == 0] = np.nan

sinX = (np.sin(np.arctan(slopeX[0, :, :-1])) + np.sin(np.arctan(slopeX[0, :, 1:]))) / 2.0
sinX = np.repeat(sinX[np.newaxis, :, :], nz, axis=0)
cosX = (np.cos(np.arctan(slopeX[0, :, :-1])) + np.cos(np.arctan(slopeX[0, :, 1:]))) / 2.0
cosX = np.repeat(cosX[np.newaxis, :, :], nz, axis=0)
sinY = (np.sin(np.arctan(slopeY[0, :-1, :])) + np.sin(np.arctan(slopeY[0, 1:, :]))) / 2.0
sinY = np.repeat(sinY[np.newaxis, :, :], nz, axis=0)
cosY = (np.cos(np.arctan(slopeY[0, :-1, :])) + np.cos(np.arctan(slopeY[0, 1:, :]))) / 2.0
cosY = np.repeat(cosY[np.newaxis, :, :], nz, axis=0)

invpermX = 1 / permX
kmeanX = 2.0 / (invpermX[:, :, :-1] + invpermX[:, :, 1:])
gradX = np.diff(pressMask, axis=2) / dx
gradX = gradX * cosX + sinX
krelX = (krel[:, :, 1:] + krel[:, :, :-1]) / 2
flowX = -kmeanX * gradX * krelX

invpermY = 1 / permY
kmeanY = 2.0 / (invpermY[:, :-1, :] + invpermY[:, 1:, :])
gradY = np.diff(pressMask, axis=1) / dy
gradY = gradY * cosY + sinY
krelY = (krel[:, 1:, :] + krel[:, :-1, :])
flowY = -kmeanY * gradY * krelY


matDZ=np.repeat(dz[:,np.newaxis,],ny,axis=1)
matDZ=np.repeat(matDZ[:,:,np.newaxis,],nx,axis=2)
kmeanZ = ( (matDZ[:-1] + matDZ[1:]) /(matDZ[:-1]/permZ[:-1,:,:] + matDZ[1:]/permZ[1:,:,:]))
gradZ = 1 + np.diff(pressMask, axis=0) * 2. / (matDZ[:-1] + matDZ[1:])
krelZ = (krel[1:, :, :] + krel[:-1, :, :])
flowZ = -kmeanZ * gradZ * krelZ

z=np.cumsum(dz)
matZ=np.repeat(z[:,np.newaxis,],ny,axis=1)
matZ=np.repeat(matZ[:,:,np.newaxis,],nx,axis=2)
gradZ=np.diff(pressMask,axis=0)


# plot each z
temp = gradZ.copy()
# temp=kmeanX
fig, ax = plt.subplots(2, 5, figsize=(18, 5))
temp[data.mask == 0] = np.nan
vmax = np.nanmax(temp)
vmin = np.nanmin(temp)
for k in range(10):
    iy, ix = utils.index2d(k, 2, 5)
    cb = ax[iy, ix].pcolor(temp[k, :, :], vmax=vmax, vmin=vmin)
    # cb = ax[iy, ix].pcolor(temp[k, :, :])
    ax[iy, ix].set_xticklabels([])
    ax[iy, ix].set_yticklabels([])
    fig.colorbar(cb)
fig.show()


temp = velz.copy()
fig, ax = plt.subplots(2, 5, figsize=(18, 5))
temp[data.mask == 0] = np.nan
vmax = np.nanmax(temp)
vmin = np.nanmin(temp)
for k in range(10):
    iy, ix = utils.index2d(k, 2, 5)
    # cb = ax[iy, ix].pcolor(temp[k, :, :], vmax=vmax, vmin=vmin)
    cb = ax[iy, ix].pcolor(temp[k, :, :])
    ax[iy, ix].set_xticklabels([])
    ax[iy, ix].set_yticklabels([])
    fig.colorbar(cb)
fig.show()

