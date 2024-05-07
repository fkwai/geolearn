import sys, os, parflow
from parflow import Run
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from hydroDL import kPath

nt = 120

# mesh
run_name = '15060202'
work_dir = os.path.join(kPath.dirParflow, run_name)
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))
data = run.data_accessor
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape
xx = np.arange(0, nx + 1) * dx
zz = np.insert(np.cumsum(dz), 0, 0)
xm, zm = np.meshgrid(xx, zz)
slopeX = data.slope_x

mask = np.array(data.mask) 
mask[mask > 0] = 1
mask = mask.astype(bool)
maskS = mask[0, :, :]

# zm[:, 1:] = zm[:, 1:] + slopeX * xm[:, 1:]


# var = 'qflx_infl'
# infl = parflow.read_pfb(
#     os.path.join(work_dir, '{}.out.{}.{:05d}.pfb'.format(run_name, var, t))
# )


file = '/home/kuai/work/parflow/LW/input/NLDAS/NLDAS.APCP.000097_to_000120.pfb'
p = parflow.read_pfb(file)
p.sum()

# total storage
f = 3.6
data.time = 0
storage = data.subsurface_storage
pond = data.surface_storage
s0 = storage.sum() + pond.sum()
mat = np.zeros([nt, 5])
temp = 0
for k in range(nt):
    data.time = k + 1
    data.forcing_time = k
    storage = data.subsurface_storage
    pond = data.surface_storage
    outflow = data.overland_flow()
    s1 = storage.sum() + pond.sum()
    evap = data.clm_output('qflx_evap_tot')
    p = data.clm_forcing('APCP')
    mat[k, 0] = s1 - s0
    mat[k, 1] = (evap * maskS).sum() * dx * dy * f
    mat[k, 2] = outflow.sum()
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



fig, ax = plt.subplots(1, 1)
ax.plot(mat[:, 2],mat[:, 0],'*-')
ax.legend()
fig.show()


file = '/home/kuai/work/parflow/LW/input/NLDAS/NLDAS.APCP.000097_to_000120.pfb'
p = parflow.read_pfb(file)
