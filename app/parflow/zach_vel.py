
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


run_name='test'

work_dir = os.path.join(kPath.dirParflow, run_name)
run = Run.from_definition(os.path.join(work_dir, run_name + '.pfidb'))


data = run.data_accessor
data.time = 10

velx = data._pfb_to_array(f'{data._name}.out.velx.{data._ts}.pfb')
vely = data._pfb_to_array(f'{data._name}.out.vely.{data._ts}.pfb')
velz = data._pfb_to_array(f'{data._name}.out.velz.{data._ts}.pfb')

fig, ax = plt.subplots(1, 1, figsize=(18, 5))
ax.pcolor(velz[:,0,:])

fig.show()