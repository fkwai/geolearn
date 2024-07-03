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

run_name = "14080103"
work_dir = os.path.join(kPath.dirParflow, run_name, 'outputs')
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))
data = run.data_accessor
dx, dy, dz = data.dx, data.dy, data.dz
nz, ny, nx = data.shape

elev = np.zeros([ny, nx])

slopeX = data.slope_x[0, :, :]
slopeY = data.slope_y[0, :, :]


for i in range(1, nx):
    elev[0, i] = elev[0, i - 1] + slopeX[0, i - 1]

for j in range(1, ny):
    elev[j, 0] = elev[j - 1, 0] + slopeX[j - 1, 0]
    for i in range(1, nx):
        elev[j, i] = elev[j, i - 1] + slopeX[j, i - 1]
        elev[j, i] += elev[j - 1, i] + slopeX[j - 1, i] - elev[j - 1, i - 1]


slopeXC = np.cumsum(slopeX, axis=1)
slopeYC = np.cumsum(slopeY, axis=0)
elev = slopeXC * dx + slopeYC * dy

parflow.write_pfb(os.path.join(work_dir, 'dem.pfb'), elev)
