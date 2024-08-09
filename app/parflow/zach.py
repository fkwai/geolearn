# Mixed boundary condition test case for EcoSLIM
# Demonstrates incorporation of ParFlow DirEquilRefPatch and 
# FluxConst boundary conditions across a hillslope

tcl_precision = 17

# Get the run name from the file name
# variable file_name [info script]
# variable end_index [expr [string length $file_name] - 5]
# variable run_name [string range $file_name 0 $end_index]
# file mkdir $run_name
# cd $run_name

#
# Import the ParFlow TCL package
#
from parflow import Run
run = Run("run", __file__)

run.FileVersion = 4

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
run.ComputationalGrid.NY = 1
run.ComputationalGrid.NZ = 40

run.ComputationalGrid.DX = 1.0
run.ComputationalGrid.DY = 1.0
run.ComputationalGrid.DZ = .1

#---------------------------------------------------------
# The Names of the GeomInputs
#---------------------------------------------------------
run.GeomInput.Names = 'domaininput'

run.GeomInput.domaininput.GeomName = 'domain'
run.GeomInput.domaininput.InputType = 'Box'

#---------------------------------------------------------
# Domain Geometry
#---------------------------------------------------------
run.Geom.domain.Lower.X = 0.0
run.Geom.domain.Lower.Y = 0.0
run.Geom.domain.Lower.Z = 0.0

run.Geom.domain.Upper.X = 30.0
run.Geom.domain.Upper.Y = 1.0
run.Geom.domain.Upper.Z = 4.0
run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'


#-----------------------------------------------------------------------------
# Perm
#-----------------------------------------------------------------------------

run.Geom.Perm.Names = 'domain'

# Values in m/hour#

run.Geom.domain.Perm.Type = 'Constant'
run.Geom.domain.Perm.Value = 0.1


run.Perm.TensorType = 'TensorByGeom'

run.Geom.Perm.TensorByGeom.Names = 'domain'

run.Geom.domain.Perm.TensorValX = 1.0d0
run.Geom.domain.Perm.TensorValY = 1.0d0
run.Geom.domain.Perm.TensorValZ = 1.0d0

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

#
run.TimingInfo.BaseUnit = 0.1
run.TimingInfo.StartCount = 0
run.TimingInfo.StartTime = 0.0
run.TimingInfo.StopTime = 200000
run.TimingInfo.DumpInterval = 10000.0

run.TimeStep.Type = 'Growth'
run.TimeStep.InitialStep = 0.1
run.TimeStep.GrowthFactor = 1.1
run.TimeStep.MaxStep = 100000000
run.TimeStep.MinStep = 0.01

#-----------------------------------------------------------------------------
# Porosity
#-----------------------------------------------------------------------------

run.Geom.Porosity.GeomNames = 'domain'

run.Geom.domain.Porosity.Type = 'Constant'
run.Geom.domain.Porosity.Value = 0.25


#-----------------------------------------------------------------------------
# Domain
#-----------------------------------------------------------------------------

run.Domain.GeomName = 'domain'

#-----------------------------------------------------------------------------
# Relative Permeability
#-----------------------------------------------------------------------------

run.Phase.RelPerm.Type = 'VanGenuchten'
run.Phase.RelPerm.GeomNames = 'domain'

run.Geom.domain.RelPerm.Alpha = 6.0
run.Geom.domain.RelPerm.N = 2.

#---------------------------------------------------------
# Saturation
#---------------------------------------------------------

run.Phase.Saturation.Type = 'VanGenuchten'
run.Phase.Saturation.GeomNames = 'domain'

run.Geom.domain.Saturation.Alpha = 6.0
run.Geom.domain.Saturation.N = 2.
run.Geom.domain.Saturation.SRes = 0.02
run.Geom.domain.Saturation.SSat = 1.0



#-----------------------------------------------------------------------------
# Wells
#-----------------------------------------------------------------------------
run.Wells.Names = ''

#-----------------------------------------------------------------------------
# Time Cycles
#-----------------------------------------------------------------------------
run.Cycle.Names = 'constant'
run.Cycle.constant.Names = 'alltime'
run.Cycle.constant.alltime.Length = 1
run.Cycle.constant.Repeat = -1


#-----------------------------------------------------------------------------
# Boundary Conditions: Pressure
#-----------------------------------------------------------------------------
run.BCPressure.PatchNames = [pfget Geom.domain.Patches]

run.Patch.x_lower.BCPressure.Type = 'DirEquilRefPatch'
run.Patch.x_lower.BCPressure.Cycle = 'constant'
run.Patch.x_lower.BCPressure.RefGeom = 'domain'
run.Patch.x_lower.BCPressure.RefPatch = 'z_lower'
run.Patch.x_lower.BCPressure.alltime.Value = 1.5

run.Patch.y_lower.BCPressure.Type = 'FluxConst'
run.Patch.y_lower.BCPressure.Cycle = 'constant'
run.Patch.y_lower.BCPressure.alltime.Value = 0.0

run.Patch.z_lower.BCPressure.Type = 'FluxConst'
run.Patch.z_lower.BCPressure.Cycle = 'constant'
run.Patch.z_lower.BCPressure.alltime.Value = 0.0

run.Patch.x_upper.BCPressure.Type = 'DirEquilRefPatch'
run.Patch.x_upper.BCPressure.Cycle = 'constant'
run.Patch.x_upper.BCPressure.RefGeom = 'domain'
run.Patch.x_upper.BCPressure.RefPatch = 'z_lower'
run.Patch.x_upper.BCPressure.alltime.Value = 3.0

run.Patch.y_upper.BCPressure.Type = 'FluxConst'
run.Patch.y_upper.BCPressure.Cycle = 'constant'
run.Patch.y_upper.BCPressure.alltime.Value = 0.0

run.Patch.z_upper.BCPressure.Type = 'FluxConst'
run.Patch.z_upper.BCPressure.Cycle = 'constant'
run.Patch.z_upper.BCPressure.alltime.Value = -0.0005

#---------------------------------------------------------
# Topo slopes in x-direction
#---------------------------------------------------------

run.TopoSlopesX.Type = 'Constant'
run.TopoSlopesX.GeomNames = 'domain'
run.TopoSlopesX.Geom.domain.Value = 0.00

#---------------------------------------------------------
# Topo slopes in y-direction
#---------------------------------------------------------


run.TopoSlopesY.Type = 'Constant'
run.TopoSlopesY.GeomNames = 'domain'
run.TopoSlopesY.Geom.domain.Value = 0.0

#---------------------------------------------------------
# Mannings coefficient
#---------------------------------------------------------

run.Mannings.Type = 'Constant'
run.Mannings.GeomNames = 'domain'
run.Mannings.Geom.domain.Value = 1.e-6

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
run.Solver.MaxIter = 25000
run.OverlandFlowDiffusive = 0


run.Solver.Nonlinear.MaxIter = 20
run.Solver.Nonlinear.ResidualTol = 1e-7
run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
run.Solver.Nonlinear.EtaValue = 0.01
run.Solver.Nonlinear.UseJacobian = False
run.Solver.Nonlinear.UseJacobian = True
run.Solver.Nonlinear.DerivativeEpsilon = 1e-8
run.Solver.Nonlinear.StepTol = 1e-20
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Linear.KrylovDimension = 20
run.Solver.Linear.MaxRestart = 2

run.Solver.Linear.Preconditioner = 'PFMG'
run.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'
run.Solver.PrintSubsurf = False
run. = 'Solver.Drop 1E_20'
run.Solver.AbsTol = 1E-7

run.Solver.WriteSiloSubsurfData = True
run.Solver.WriteSiloPressure = True
run.Solver.WriteSiloSaturation = True
run.Solver.WriteSiloEvapTrans = True


######
## Make sure we write PFB output for EcoSLIM
#
run.Solver.PrintVelocities = True
run.Solver.PrintEvapTrans = True



#---------------------------------------------------------
# Initial conditions: water pressure
#---------------------------------------------------------

# set water table to be at the bottom of the domain, the top layer is initially dry
run.ICPressure.Type = 'HydroStaticPatch'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.Value = 2.25

run.Geom.domain.ICPressure.RefGeom = 'domain'
run.Geom.domain.ICPressure.RefPatch = 'z_lower'

#-----------------------------------------------------------------------------
# Run and Unload the ParFlow output files
#-----------------------------------------------------------------------------


# pfrun $run_name
# pfundist $run_name

# cd ..run.run()
