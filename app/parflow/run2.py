import os
import matplotlib.pyplot as plt
import numpy as np
from parflow import Run
from parflow.tools.io import read_pfb, read_clm
from parflow.tools.fs import mkdir
from parflow.tools.settings import set_working_directory
import subsettools as st
import hf_hydrodata as hf
from hydroDL import kPath

# make huc an arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("huc", help="HUC code")
    args = parser.parse_args()
    huc = args.huc


huc = '11140102'

work_dir = os.path.join(kPath.dirParflow, huc, 'outputs')
run = Run.from_definition(os.path.join(work_dir, huc + '.pfidb'))
run.TimingInfo.StopTime = 365 * 24
run.Solver.CLM.MetFileName = "CW3E"
run.Solver.PrintCLM = True
run.Solver.PrintVelocities = True
run.Solver.PrintMannings = True
run.Solver.PrintSubsurfData = True
run.ComputationalGrid.NZ = 10

run.ComputationalGrid.DZ

data = run.data_accessor
test = data._pfb_to_array('pme.pfb')

fig, ax = plt.subplots(1,1, figsize=(18, 5))
ax.imshow(test[0,:,:])
fig.show()


/home/kuai/work/parflow/11140102/outputs

# run.run(working_directory=work_dir)
