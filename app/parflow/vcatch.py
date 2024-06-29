import sys
import os
import numpy as np
from parflow import Run
from parflow.tools.fs import mkdir, get_absolute_path
from parflow.tools.io import read_pfb, write_pfb, ParflowBinaryReader
from parflow.tools.top import compute_top
from parflow.tools import hydrology
from parflow.tools.compare import pf_test_equal
from hydroDL import kPath

run_name = "vcatch"
work_dir = os.path.join(kPath.dirParflow, run_name)
input_dir = os.path.join(work_dir, "input")
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

run = Run(run_name)
nt = 120

#---------------------------------------------------------
# Flux on the top surface
#---------------------------------------------------------

rain_flux = -0.05
rec_flux = 0.0

#---------------------------------------------------------

run.FileVersion = 4

#---------------------------------------------------------
# Processor topology
#---------------------------------------------------------

run.Process.Topology.P = 1
run.Process.Topology.Q = 1
run.Process.Topology.R = 1

#---------------------------------------------------------
# Computational Grid
#---------------------------------------------------------

run.ComputationalGrid.Lower.X = 0.0
run.ComputationalGrid.Lower.Y = 0.0
run.ComputationalGrid.Lower.Z = 0.0

run.ComputationalGrid.NX = 30
run.ComputationalGrid.NY = 30
run.ComputationalGrid.NZ = 30

run.ComputationalGrid.DX = 10.0
run.ComputationalGrid.DY = 10.0
run.ComputationalGrid.DZ = 0.05

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------

run.GeomInput.Names = 'domaininput leftinput rightinput channelinput'

run.GeomInput.domaininput.GeomName = 'domain'
run.GeomInput.leftinput.GeomName = 'left'
run.GeomInput.rightinput.GeomName = 'right'
run.GeomInput.channelinput.GeomName = 'channel'

run.GeomInput.domaininput.InputType = 'Box'
run.GeomInput.leftinput.InputType = 'Box'
run.GeomInput.rightinput.InputType = 'Box'
run.GeomInput.channelinput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry 
#---------------------------------------------------------

run.Geom.domain.Lower.X = 0.0
run.Geom.domain.Lower.Y = 0.0
run.Geom.domain.Lower.Z = 0.0
run.Geom.domain.Upper.X = 300.0
run.Geom.domain.Upper.Y = 300.0
run.Geom.domain.Upper.Z = 1.5
run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

#---------------------------------------------------------
# Left Slope Geometry 
#---------------------------------------------------------

run.Geom.left.Lower.X = 0.0
run.Geom.left.Lower.Y = 0.0
run.Geom.left.Lower.Z = 0.0
run.Geom.left.Upper.X = 300.0
run.Geom.left.Upper.Y = 140.0
run.Geom.left.Upper.Z = 1.5

#---------------------------------------------------------
# Right Slope Geometry 
#---------------------------------------------------------

run.Geom.right.Lower.X = 0.0
run.Geom.right.Lower.Y = 160.0
run.Geom.right.Lower.Z = 0.0
run.Geom.right.Upper.X = 300.0
run.Geom.right.Upper.Y = 300.0
run.Geom.right.Upper.Z = 1.5

#---------------------------------------------------------
# Channel Geometry 
#---------------------------------------------------------

run.Geom.channel.Lower.X = 0.0
run.Geom.channel.Lower.Y = 140.0
run.Geom.channel.Lower.Z = 0.0
run.Geom.channel.Upper.X = 300.0
run.Geom.channel.Upper.Y = 160.0
run.Geom.channel.Upper.Z = 1.5

#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

run.Geom.Perm.Names = 'left right channel'

# Values in m/hour

# these are examples to make the upper portions of the v heterogeneous
# the following is ignored if the perm.type "Constant" settings are not
# commented out, below.

run.Geom.left.Perm.Type = 'TurnBands'
run.Geom.left.Perm.LambdaX = 50.
run.Geom.left.Perm.LambdaY = 50.
run.Geom.left.Perm.LambdaZ = 0.5
run.Geom.left.Perm.GeomMean = 0.01

run.Geom.left.Perm.Sigma = 0.5
run.Geom.left.Perm.NumLines = 40
run.Geom.left.Perm.RZeta = 5.0
run.Geom.left.Perm.KMax = 100.0
run.Geom.left.Perm.DelK = 0.2
run.Geom.left.Perm.Seed = 33333
run.Geom.left.Perm.LogNormal = 'Log'
run.Geom.left.Perm.StratType = 'Bottom'

run.Geom.right.Perm.Type = 'TurnBands'
run.Geom.right.Perm.LambdaX = 50.
run.Geom.right.Perm.LambdaY = 50.
run.Geom.right.Perm.LambdaZ = 0.5
run.Geom.right.Perm.GeomMean = 0.05

run.Geom.right.Perm.Sigma = 0.5
run.Geom.right.Perm.NumLines = 40
run.Geom.right.Perm.RZeta = 5.0
run.Geom.right.Perm.KMax = 100.0
run.Geom.right.Perm.DelK = 0.2
run.Geom.right.Perm.Seed = 13333
run.Geom.right.Perm.LogNormal = 'Log'
run.Geom.right.Perm.StratType = 'Bottom'

# hydraulic conductivity is very low, but not zero, top node will have to saturate
# before overland flow can begin and will be driven by hortonian flow
# comment out the left and right settings to make the subsurface heterogeneous using
# turning bands above.  Run time increases quite a bit with a heterogeneous
# subsurface

run.Geom.left.Perm.Type = 'Constant'
run.Geom.left.Perm.Value = 0.001

run.Geom.right.Perm.Type = 'Constant'
run.Geom.right.Perm.Value = 0.01

run.Geom.channel.Perm.Type = 'Constant'
run.Geom.channel.Perm.Value = 0.00001

run.Perm.TensorType = 'TensorByGeom'

run.Geom.Perm.TensorByGeom.Names = 'domain'

run.Geom.domain.Perm.TensorValX = 1.0
run.Geom.domain.Perm.TensorValY = 1.0
run.Geom.domain.Perm.TensorValZ = 1.0

#-----------------------------------------------------------------------------
# Specific Storage
#-----------------------------------------------------------------------------

run.SpecificStorage.Type = 'Constant'
run.SpecificStorage.GeomNames = 'domain'
run.Geom.domain.SpecificStorage.Value = 1.0e-4

#-----------------------------------------------------------------------------
# Phases
#-----------------------------------------------------------------------------

run.Phase.Names = 'water'

run.Phase.water.Density.Type = 'Constant'
run.Phase.water.Density.Value = 1.0

run.Phase.water.Viscosity.Type = 'Constant'
run.Phase.water.Viscosity.Value = 1.0

#-----------------------------------------------------------------------------
# Contaminants
#-----------------------------------------------------------------------------

run.Contaminants.Names = ''

#-----------------------------------------------------------------------------
# Retardation
#-----------------------------------------------------------------------------

run.Geom.Retardation.GeomNames = ''

#-----------------------------------------------------------------------------
# Gravity
#-----------------------------------------------------------------------------

run.Gravity = 1.0

#-----------------------------------------------------------------------------
# Setup timing info
#-----------------------------------------------------------------------------

run.TimingInfo.BaseUnit = 0.1
run.TimingInfo.StartCount = 0
run.TimingInfo.StartTime = 0.0
run.TimingInfo.StopTime = 2.0
run.TimingInfo.DumpInterval = 0.1
run.TimeStep.Type = 'Constant'
run.TimeStep.Value = 0.1

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

run.Geom.Porosity.GeomNames = 'left right channel'

run.Geom.left.Porosity.Type = 'Constant'
run.Geom.left.Porosity.Value = 0.25

run.Geom.right.Porosity.Type = 'Constant'
run.Geom.right.Porosity.Value = 0.25

run.Geom.channel.Porosity.Type = 'Constant'
run.Geom.channel.Porosity.Value = 0.01

#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

run.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

run.Phase.RelPerm.Type = 'VanGenuchten'
run.Phase.RelPerm.GeomNames = 'domain'

run.Geom.domain.RelPerm.Alpha = 0.5
run.Geom.domain.RelPerm.N = 3.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

run.Phase.Saturation.Type = 'VanGenuchten'
run.Phase.Saturation.GeomNames = 'domain'

run.Geom.domain.Saturation.Alpha = 0.5
run.Geom.domain.Saturation.N = 3.
run.Geom.domain.Saturation.SRes = 0.2
run.Geom.domain.Saturation.SSat = 1.0

#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------

run.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------

run.Cycle.Names = 'constant rainrec'
run.Cycle.constant.Names = 'alltime'
run.Cycle.constant.alltime.Length = 1
run.Cycle.constant.Repeat = -1

# rainfall and recession time periods are defined here
# rain for 1 hour, recession for 2 hours

run.Cycle.rainrec.Names = 'r0 r1 r2 r3 r4 r5 r6'
run.Cycle.rainrec.r0.Length = 1
run.Cycle.rainrec.r1.Length = 1
run.Cycle.rainrec.r2.Length = 1
run.Cycle.rainrec.r3.Length = 1
run.Cycle.rainrec.r4.Length = 1
run.Cycle.rainrec.r5.Length = 1
run.Cycle.rainrec.r6.Length = 1

run.Cycle.rainrec.Repeat = 1

#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------

run.BCPressure.PatchNames = run.Geom.domain.Patches

run.Patch.x_lower.BCPressure.Type = 'FluxConst'
run.Patch.x_lower.BCPressure.Cycle = 'constant'
run.Patch.x_lower.BCPressure.alltime.Value = 0.0

run.Patch.y_lower.BCPressure.Type = 'FluxConst'
run.Patch.y_lower.BCPressure.Cycle = 'constant'
run.Patch.y_lower.BCPressure.alltime.Value = 0.0

run.Patch.z_lower.BCPressure.Type = 'FluxConst'
run.Patch.z_lower.BCPressure.Cycle = 'constant'
run.Patch.z_lower.BCPressure.alltime.Value = 0.0

run.Patch.x_upper.BCPressure.Type = 'FluxConst'
run.Patch.x_upper.BCPressure.Cycle = 'constant'
run.Patch.x_upper.BCPressure.alltime.Value = 0.0

run.Patch.y_upper.BCPressure.Type = 'FluxConst'
run.Patch.y_upper.BCPressure.Cycle = 'constant'
run.Patch.y_upper.BCPressure.alltime.Value = 0.0


run.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
run.Patch.z_upper.BCPressure.Cycle = 'rainrec'
run.Patch.z_upper.BCPressure.r0.Value = rec_flux
run.Patch.z_upper.BCPressure.r1.Value = rec_flux
run.Patch.z_upper.BCPressure.r2.Value = rain_flux
run.Patch.z_upper.BCPressure.r3.Value = rain_flux
run.Patch.z_upper.BCPressure.r4.Value = rec_flux
run.Patch.z_upper.BCPressure.r5.Value = rec_flux
run.Patch.z_upper.BCPressure.r6.Value = rec_flux

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

run.TopoSlopesX.Type = 'Constant'
run.TopoSlopesX.GeomNames = 'left right channel'
if use_slopes > 0:
  run.TopoSlopesX.Geom.left.Value = 0.000
  run.TopoSlopesX.Geom.right.Value = 0.000
  run.TopoSlopesX.Geom.channel.Value = 0.001*use_slopes
else:
  run.TopoSlopesX.Geom.left.Value = 0.000
  run.TopoSlopesX.Geom.right.Value = 0.000
  run.TopoSlopesX.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------

run.TopoSlopesY.Type = 'Constant'
run.TopoSlopesY.GeomNames = 'left right channel'
if use_slopes > 0:
  run.TopoSlopesY.Geom.left.Value = -0.005
  run.TopoSlopesY.Geom.right.Value = 0.005
  run.TopoSlopesY.Geom.channel.Value = 0.000
else:
  run.TopoSlopesY.Geom.left.Value = 0.000
  run.TopoSlopesY.Geom.right.Value = 0.000
  run.TopoSlopesY.Geom.channel.Value = 0.000

#---------------------------------------------------------
# Mannings coefficient 
#---------------------------------------------------------

run.Mannings.Type = 'Constant'
run.Mannings.GeomNames = 'left right channel'
run.Mannings.Geom.left.Value = 5.e-6
run.Mannings.Geom.right.Value = 5.e-6
run.Mannings.Geom.channel.Value = 1.e-6

#-----------------------------------------------------------------------------
# Phase sources:
#-----------------------------------------------------------------------------

run.PhaseSources.water.Type = 'Constant'
run.PhaseSources.water.GeomNames = 'domain'
run.PhaseSources.water.Geom.domain.Value = 0.0

#-----------------------------------------------------------------------------
# Exact solution specification for error calculations
#-----------------------------------------------------------------------------

run.KnownSolution = 'NoKnownSolution'

#-----------------------------------------------------------------------------
# Set solver parameters
#-----------------------------------------------------------------------------

run.Solver = 'Richards'
run.Solver.MaxIter = 100

run.Solver.AbsTol = 1E-10
run.Solver.Nonlinear.MaxIter = 20
run.Solver.Nonlinear.ResidualTol = 1e-9
run.Solver.Nonlinear.EtaChoice = 'Walker1'
run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
run.Solver.Nonlinear.EtaValue = 0.01
run.Solver.Nonlinear.UseJacobian = False
run.Solver.Nonlinear.DerivativeEpsilon = 1e-8
run.Solver.Nonlinear.StepTol = 1e-30
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Linear.KrylovDimension = 20
run.Solver.Linear.MaxRestart = 2

run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PFMG.MaxIter = 1
run.Solver.Linear.Preconditioner.PFMG.Smoother = 'RBGaussSeidelNonSymmetric'
run.Solver.Linear.Preconditioner.PFMG.NumPreRelax = 1
run.Solver.Linear.Preconditioner.PFMG.NumPostRelax = 1

run.Solver.PrintSubsurfData = True
run.Solver.PrintConcentration = True
run.Solver.PrintSlopes = True
run.Solver.PrintEvapTrans = True
run.Solver.PrintEvapTransSum = True
run.Solver.PrintOverlandSum = True
run.Solver.PrintMannings = True
run.Solver.PrintSpecificStorage = True

#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
run.ICPressure.Type = 'HydroStaticPatch'
run.ICPressure.GeomNames = 'domain'

run.Geom.domain.ICPressure.Value = -3.0

run.Geom.domain.ICPressure.RefGeom = 'domain'
run.Geom.domain.ICPressure.RefPatch = 'z_upper'
