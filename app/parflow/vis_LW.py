import sys, os, parflow
from parflow import Run
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from hydroDL import kPath

nt = 120

# mesh
run_name = "LW"
work_dir = os.path.join(kPath.dirParflow, run_name)
run = Run.from_definition(os.path.join(work_dir, run_name + ".pfidb"))
data = run.data_accessor
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape
xx = np.arange(0, nx + 1) * dx
zz = np.insert(np.cumsum(dz), 0, 0)
xm, zm = np.meshgrid(xx, zz)
slopeX = data.slope_x
# zm[:, 1:] = zm[:, 1:] + slopeX * xm[:, 1:]

cmap = matplotlib.cm.get_cmap("jet")
fig, ax = plt.subplots(1, 1)
hLst = list()
for t in range(0, nt + 1):
    data.time = t
    h = data.wtd
    hLst.append(h.mean())
ax.plot(hLst)
ax.legend()
fig.show()

data.time = 0
fig, ax = plt.subplots(1, 1)
cb = ax.imshow(h)
fig.colorbar(cb)

fig.show()

tRange = np.arange(nt)
sLst = list()
s2Lst = list()
for t in tRange:
    data.time = t
    storage = data.subsurface_storage
    sLst.append(np.mean(storage))
    outflow = data.overland_flow_grid()

    # data.subsurface_storage
    # porosity = data.computed_porosity
    # specific_storage = data.specific_storage
    # mask = data.mask
    # pressure = data.pressure
    # saturation = data.saturation
    # dx,dy,dz=data.dx,data.dy,data.dz
    # dz = dz[:, np.newaxis, np.newaxis]
    # incompressible = porosity * saturation * dz * dx * dy
    # compressible = pressure * saturation * specific_storage * dz * dx * dy
    # storage=incompressible+compressible
    # sLst.append(np.mean(storage))
    # saturation2=saturation.copy()
    # saturation2[saturation2<1]=0
    # incompressible2 = porosity * saturation2 * dz * dx * dy
    # compressible2 = pressure * saturation2 * specific_storage * dz * dx * dy
    # storage2=incompressible2+compressible2
    # s2Lst.append(np.mean(storage2))


s = np.array(sLst)
fig, ax = plt.subplots(1, 1)
ds = s[1:] - s[:-1]
ax.plot(s[:-1], ds, "*")
fig.show()
