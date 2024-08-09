import sys, os, parflow
from parflow import Run
from parflow.tools.fs import mkdir, cp, chdir, get_absolute_path
import os
from hydroDL import kPath
import shutil
import numpy

run_name = 'LW2'
work_dir = os.path.join(kPath.dirParflow, run_name)

run = Run(run_name)
nt = 120
nc = 3
# -----------------------------------------------------------------------------
# Make a directory for the simulation run and copy files
# -----------------------------------------------------------------------------
# copy directory
input_dir = os.path.join(work_dir, 'input')
if not os.path.exists(input_dir):
    shutil.copytree(os.path.join(kPath.dirParflow, 'LW', 'input'), input_dir)

# # sh=1,25,49 ,,,
# # eh=24,48,72 ,,,

# sh=list(range(1, nt, 24))
# eh=list(range(24, nt + 1, 24))
# pLst=list()
# for k in range(len(sh)):
#     p = parflow.read_pfb(
#         os.path.join(input_dir,'NLDAS', 'NLDAS.APCP.{:06d}_to_{:06d}.pfb'.format(sh[k], eh[k]))
#     )
#     temp=p.sum(axis=(1,2))
#     pLst.append(temp)
# fig,ax=plt.subplots(1,1)
# ax.plot(np.concatenate(pLst))
# fig.show()


file_indi = os.path.join(input_dir, 'parflow_input', 'IndicatorFile_Gleeson.50z.pfb')
file_slopex = os.path.join(input_dir, 'parflow_input', 'LW.slopex.pfb')
file_slopey = os.path.join(input_dir, 'parflow_input', 'LW.slopey.pfb')
file_initP = os.path.join(input_dir, 'parflow_input', 'press.init.pfb')

run.dist(file_indi)
run.dist(file_slopex)
run.dist(file_slopey)
run.dist(file_initP)

dir_clm = os.path.join(input_dir, 'clm_input')
cp(os.path.join(dir_clm, 'drv_clmin.dat'), work_dir)
cp(os.path.join(dir_clm, 'drv_vegm.alluv.dat'), work_dir)
cp(os.path.join(dir_clm, 'drv_vegp.dat'), work_dir)

dir_met = os.path.join(input_dir, 'NLDAS')

nldas_files = list()
ind1 = list(range(1, 120, 24))
ind2 = list(range(24, 120 + 1, 24))
var_nldas = ['DSWR', 'DLWR', 'APCP', 'Temp', 'UGRD', 'VGRD', 'Press', 'SPFH']
for i1, i2 in zip(ind1, ind2):
    for var in var_nldas:
        file_nldas = 'NLDAS.{}.{:06d}_to_{:06d}.pfb'.format(var, i1, i2)
        nldas_files.append(file_nldas)
        for i3 in range(1, nc):
            file_nldas_cp = 'NLDAS.{}.{:06d}_to_{:06d}.pfb'.format(var, i1 + i3 * nt, i2 + i3 * nt)
            cp(os.path.join(dir_met, file_nldas), os.path.join(dir_met, file_nldas_cp))
            nldas_files.append(file_nldas_cp)

for file in nldas_files:
    run.dist(os.path.join(dir_met, file))

# -----------------------------------------------------------------------------

run.FileVersion = 4

# -----------------------------------------------------------------------------
# Set Processor topology
# -----------------------------------------------------------------------------
run.Process.Topology.P = 1
run.Process.Topology.Q = 1
run.Process.Topology.R = 1

# -----------------------------------------------------------------------------
# Computational Grid
# -----------------------------------------------------------------------------

run.ComputationalGrid.Lower.X = 0.0
run.ComputationalGrid.Lower.Y = 0.0
run.ComputationalGrid.Lower.Z = 0.0

run.ComputationalGrid.DX = 1000.0
run.ComputationalGrid.DY = 1000.0
run.ComputationalGrid.DZ = 2.0

run.ComputationalGrid.NX = 41
run.ComputationalGrid.NY = 41
run.ComputationalGrid.NZ = 50

# -----------------------------------------------------------------------------
# Names of the GeomInputs
# -----------------------------------------------------------------------------

run.GeomInput.Names = 'box_input indi_input'

# -----------------------------------------------------------------------------
# Domain Geometry Input
# -----------------------------------------------------------------------------

run.GeomInput.box_input.InputType = 'Box'
run.GeomInput.box_input.GeomName = 'domain'

# -----------------------------------------------------------------------------
# Domain Geometry
# -----------------------------------------------------------------------------

run.Geom.domain.Lower.X = 0.0
run.Geom.domain.Lower.Y = 0.0
run.Geom.domain.Lower.Z = 0.0
#
run.Geom.domain.Upper.X = 41000.0
run.Geom.domain.Upper.Y = 41000.0
run.Geom.domain.Upper.Z = 100.0
run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

# -----------------------------------------------------------------------------
# Indicator Geometry Input
# -----------------------------------------------------------------------------

run.GeomInput.indi_input.InputType = 'IndicatorField'
run.GeomInput.indi_input.GeomNames = 's1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 g1 g2 g3 g4 g5 g6 g7 g8'
run.Geom.indi_input.FileName = file_indi
run.GeomInput.s1.Value = 1
run.GeomInput.s2.Value = 2
run.GeomInput.s3.Value = 3
run.GeomInput.s4.Value = 4
run.GeomInput.s5.Value = 5
run.GeomInput.s6.Value = 6
run.GeomInput.s7.Value = 7
run.GeomInput.s8.Value = 8
run.GeomInput.s9.Value = 9
run.GeomInput.s10.Value = 10
run.GeomInput.s11.Value = 11
run.GeomInput.s12.Value = 12
run.GeomInput.s13.Value = 13
run.GeomInput.g1.Value = 21
run.GeomInput.g2.Value = 22
run.GeomInput.g3.Value = 23
run.GeomInput.g4.Value = 24
run.GeomInput.g5.Value = 25
run.GeomInput.g6.Value = 26
run.GeomInput.g7.Value = 27
run.GeomInput.g8.Value = 28

# -----------------------------------------------------------------------------
# Permeability (values in m/hr)
# -----------------------------------------------------------------------------

run.Geom.Perm.Names = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 g2 g3 g6 g8'

run.Geom.domain.Perm.Type = 'Constant'
run.Geom.domain.Perm.Value = 0.2

run.Geom.s1.Perm.Type = 'Constant'
run.Geom.s1.Perm.Value = 0.269022595

run.Geom.s2.Perm.Type = 'Constant'
run.Geom.s2.Perm.Value = 0.043630356

run.Geom.s3.Perm.Type = 'Constant'
run.Geom.s3.Perm.Value = 0.015841225

run.Geom.s4.Perm.Type = 'Constant'
run.Geom.s4.Perm.Value = 0.007582087

run.Geom.s5.Perm.Type = 'Constant'
run.Geom.s5.Perm.Value = 0.01818816

run.Geom.s6.Perm.Type = 'Constant'
run.Geom.s6.Perm.Value = 0.005009435

run.Geom.s7.Perm.Type = 'Constant'
run.Geom.s7.Perm.Value = 0.005492736

run.Geom.s8.Perm.Type = 'Constant'
run.Geom.s8.Perm.Value = 0.004675077

run.Geom.s9.Perm.Type = 'Constant'
run.Geom.s9.Perm.Value = 0.003386794

run.Geom.g2.Perm.Type = 'Constant'
run.Geom.g2.Perm.Value = 0.025

run.Geom.g3.Perm.Type = 'Constant'
run.Geom.g3.Perm.Value = 0.059

run.Geom.g6.Perm.Type = 'Constant'
run.Geom.g6.Perm.Value = 0.2

run.Geom.g8.Perm.Type = 'Constant'
run.Geom.g8.Perm.Value = 0.68

run.Perm.TensorType = 'TensorByGeom'
run.Geom.Perm.TensorByGeom.Names = 'domain'
run.Geom.domain.Perm.TensorValX = 1.0
run.Geom.domain.Perm.TensorValY = 1.0
run.Geom.domain.Perm.TensorValZ = 1.0

# -----------------------------------------------------------------------------
# Specific Storage
# -----------------------------------------------------------------------------

run.SpecificStorage.Type = 'Constant'
run.SpecificStorage.GeomNames = 'domain'
run.Geom.domain.SpecificStorage.Value = 1.0e-5

# -----------------------------------------------------------------------------
# Phases
# -----------------------------------------------------------------------------

run.Phase.Names = 'water'
run.Phase.water.Density.Type = 'Constant'
run.Phase.water.Density.Value = 1.0
run.Phase.water.Viscosity.Type = 'Constant'
run.Phase.water.Viscosity.Value = 1.0

# -----------------------------------------------------------------------------
# Contaminants
# -----------------------------------------------------------------------------

run.Contaminants.Names = ''

# -----------------------------------------------------------------------------
# Gravity
# -----------------------------------------------------------------------------

run.Gravity = 1.0

# -----------------------------------------------------------------------------
# Timing (time units is set by units of permeability)
# -----------------------------------------------------------------------------

run.TimingInfo.BaseUnit = 1.0
run.TimingInfo.StartCount = 0
run.TimingInfo.StartTime = 0.0
run.TimingInfo.StopTime = nt * nc
run.TimingInfo.DumpInterval = 1.0
run.TimeStep.Type = 'Constant'
run.TimeStep.Value = 1.0

# -----------------------------------------------------------------------------
# Porosity
# -----------------------------------------------------------------------------

run.Geom.Porosity.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9'

run.Geom.domain.Porosity.Type = 'Constant'
run.Geom.domain.Porosity.Value = 0.4

run.Geom.s1.Porosity.Type = 'Constant'
run.Geom.s1.Porosity.Value = 0.375

run.Geom.s2.Porosity.Type = 'Constant'
run.Geom.s2.Porosity.Value = 0.39

run.Geom.s3.Porosity.Type = 'Constant'
run.Geom.s3.Porosity.Value = 0.387

run.Geom.s4.Porosity.Type = 'Constant'
run.Geom.s4.Porosity.Value = 0.439

run.Geom.s5.Porosity.Type = 'Constant'
run.Geom.s5.Porosity.Value = 0.489

run.Geom.s6.Porosity.Type = 'Constant'
run.Geom.s6.Porosity.Value = 0.399

run.Geom.s7.Porosity.Type = 'Constant'
run.Geom.s7.Porosity.Value = 0.384

run.Geom.s8.Porosity.Type = 'Constant'
run.Geom.s8.Porosity.Value = 0.482

run.Geom.s9.Porosity.Type = 'Constant'
run.Geom.s9.Porosity.Value = 0.442

# -----------------------------------------------------------------------------
# Domain
# -----------------------------------------------------------------------------

run.Domain.GeomName = 'domain'

# ----------------------------------------------------------------------------
# Mobility
# ----------------------------------------------------------------------------

run.Phase.water.Mobility.Type = 'Constant'
run.Phase.water.Mobility.Value = 1.0

# -----------------------------------------------------------------------------
# Wells
# -----------------------------------------------------------------------------

run.Wells.Names = ''

# -----------------------------------------------------------------------------
# Time Cycles
# -----------------------------------------------------------------------------

run.Cycle.Names = 'constant'
run.Cycle.constant.Names = 'alltime'
run.Cycle.constant.alltime.Length = 1
run.Cycle.constant.Repeat = nc

# -----------------------------------------------------------------------------
# Boundary Conditions
# -----------------------------------------------------------------------------

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
run.Patch.z_upper.BCPressure.Cycle = 'constant'
run.Patch.z_upper.BCPressure.alltime.Value = 0.0

# -----------------------------------------------------------------------------
# Topo slopes in x-direction
# -----------------------------------------------------------------------------

run.TopoSlopesX.Type = 'PFBFile'
run.TopoSlopesX.GeomNames = 'domain'
run.TopoSlopesX.FileName = file_slopex

# -----------------------------------------------------------------------------
# Topo slopes in y-direction
# -----------------------------------------------------------------------------

run.TopoSlopesY.Type = 'PFBFile'
run.TopoSlopesY.GeomNames = 'domain'
run.TopoSlopesY.FileName = file_slopey

# -----------------------------------------------------------------------------
# Mannings coefficient
# -----------------------------------------------------------------------------

run.Mannings.Type = 'Constant'
run.Mannings.GeomNames = 'domain'
run.Mannings.Geom.domain.Value = 5.52e-6

# -----------------------------------------------------------------------------
# Relative Permeability
# -----------------------------------------------------------------------------

run.Phase.RelPerm.Type = 'VanGenuchten'
run.Phase.RelPerm.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 '

run.Geom.domain.RelPerm.Alpha = 3.5
run.Geom.domain.RelPerm.N = 2.0

run.Geom.s1.RelPerm.Alpha = 3.548
run.Geom.s1.RelPerm.N = 4.162

run.Geom.s2.RelPerm.Alpha = 3.467
run.Geom.s2.RelPerm.N = 2.738

run.Geom.s3.RelPerm.Alpha = 2.692
run.Geom.s3.RelPerm.N = 2.445

run.Geom.s4.RelPerm.Alpha = 0.501
run.Geom.s4.RelPerm.N = 2.659

run.Geom.s5.RelPerm.Alpha = 0.661
run.Geom.s5.RelPerm.N = 2.659

run.Geom.s6.RelPerm.Alpha = 1.122
run.Geom.s6.RelPerm.N = 2.479

run.Geom.s7.RelPerm.Alpha = 2.089
run.Geom.s7.RelPerm.N = 2.318

run.Geom.s8.RelPerm.Alpha = 0.832
run.Geom.s8.RelPerm.N = 2.514

run.Geom.s9.RelPerm.Alpha = 1.585
run.Geom.s9.RelPerm.N = 2.413

# -----------------------------------------------------------------------------
# Saturation
# -----------------------------------------------------------------------------

run.Phase.Saturation.Type = 'VanGenuchten'
run.Phase.Saturation.GeomNames = 'domain s1 s2 s3 s4 s5 s6 s7 s8 s9 '

run.Geom.domain.Saturation.Alpha = 3.5
run.Geom.domain.Saturation.N = 2.0
run.Geom.domain.Saturation.SRes = 0.2
run.Geom.domain.Saturation.SSat = 1.0

run.Geom.s1.Saturation.Alpha = 3.548
run.Geom.s1.Saturation.N = 4.162
run.Geom.s1.Saturation.SRes = 0.000001
run.Geom.s1.Saturation.SSat = 1.0

run.Geom.s2.Saturation.Alpha = 3.467
run.Geom.s2.Saturation.N = 2.738
run.Geom.s2.Saturation.SRes = 0.000001
run.Geom.s2.Saturation.SSat = 1.0

run.Geom.s3.Saturation.Alpha = 2.692
run.Geom.s3.Saturation.N = 2.445
run.Geom.s3.Saturation.SRes = 0.000001
run.Geom.s3.Saturation.SSat = 1.0

run.Geom.s4.Saturation.Alpha = 0.501
run.Geom.s4.Saturation.N = 2.659
run.Geom.s4.Saturation.SRes = 0.000001
run.Geom.s4.Saturation.SSat = 1.0

run.Geom.s5.Saturation.Alpha = 0.661
run.Geom.s5.Saturation.N = 2.659
run.Geom.s5.Saturation.SRes = 0.000001
run.Geom.s5.Saturation.SSat = 1.0

run.Geom.s6.Saturation.Alpha = 1.122
run.Geom.s6.Saturation.N = 2.479
run.Geom.s6.Saturation.SRes = 0.000001
run.Geom.s6.Saturation.SSat = 1.0

run.Geom.s7.Saturation.Alpha = 2.089
run.Geom.s7.Saturation.N = 2.318
run.Geom.s7.Saturation.SRes = 0.000001
run.Geom.s7.Saturation.SSat = 1.0

run.Geom.s8.Saturation.Alpha = 0.832
run.Geom.s8.Saturation.N = 2.514
run.Geom.s8.Saturation.SRes = 0.000001
run.Geom.s8.Saturation.SSat = 1.0

run.Geom.s9.Saturation.Alpha = 1.585
run.Geom.s9.Saturation.N = 2.413
run.Geom.s9.Saturation.SRes = 0.000001
run.Geom.s9.Saturation.SSat = 1.0

# -----------------------------------------------------------------------------
# Phase sources:
# -----------------------------------------------------------------------------

run.PhaseSources.water.Type = 'Constant'
run.PhaseSources.water.GeomNames = 'domain'
run.PhaseSources.water.Geom.domain.Value = 0.0

# ----------------------------------------------------------------
# CLM Settings:
# ------------------------------------------------------------

run.Solver.LSM = 'CLM'
run.Solver.CLM.CLMFileDir = dir_clm
run.Solver.CLM.Print1dOut = False
run.Solver.CLM.CLMDumpInterval = 1

run.Solver.CLM.MetForcing = '3D'
run.Solver.CLM.MetFileName = 'NLDAS'
run.Solver.CLM.MetFilePath = dir_met
run.Solver.CLM.MetFileNT = 24
run.Solver.CLM.IstepStart = 1

run.Solver.CLM.EvapBeta = 'Linear'
run.Solver.CLM.VegWaterStress = 'Saturation'
run.Solver.CLM.ResSat = 0.1
run.Solver.CLM.WiltingPoint = 0.12
run.Solver.CLM.FieldCapacity = 0.98
run.Solver.CLM.IrrigationType = 'none'

# ---------------------------------------------------------
# Initial conditions: water pressure
# ---------------------------------------------------------

run.ICPressure.Type = 'PFBFile'
run.ICPressure.GeomNames = 'domain'
run.Geom.domain.ICPressure.RefPatch = 'z_upper'
run.Geom.domain.ICPressure.FileName = file_initP

# -------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

# Writing output (all pfb):
run.Solver.PrintSubsurfData = True
run.Solver.PrintPressure = True
run.Solver.PrintSaturation = True
run.Solver.PrintMask = True
run.Solver.PrintVelocities = True
run.Solver.PrintMannings = True

run.Solver.WriteCLMBinary = False
run.Solver.PrintCLM = True
run.Solver.WriteSiloSpecificStorage = False
run.Solver.WriteSiloMannings = False
run.Solver.WriteSiloMask = False
run.Solver.WriteSiloSlopes = False
run.Solver.WriteSiloSubsurfData = False
run.Solver.WriteSiloPressure = False
run.Solver.WriteSiloSaturation = False
run.Solver.WriteSiloEvapTrans = False
run.Solver.WriteSiloEvapTransSum = False
run.Solver.WriteSiloOverlandSum = False
run.Solver.WriteSiloCLM = False
run.Solver.PrintVelocities = True

# -----------------------------------------------------------------------------
# Exact solution specification for error calculations
# -----------------------------------------------------------------------------

run.KnownSolution = 'NoKnownSolution'

# -----------------------------------------------------------------------------
# Set solver parameters
# -----------------------------------------------------------------------------

# ParFlow Solution
run.Solver = 'Richards'
run.Solver.TerrainFollowingGrid = True
run.Solver.Nonlinear.VariableDz = False

run.Solver.MaxIter = 25000
run.Solver.Drop = 1e-20
run.Solver.AbsTol = 1e-8
run.Solver.MaxConvergenceFailures = 8
run.Solver.Nonlinear.MaxIter = 80
run.Solver.Nonlinear.ResidualTol = 1e-6

## new solver settings for Terrain Following Grid
run.Solver.Nonlinear.EtaChoice = 'EtaConstant'
run.Solver.Nonlinear.EtaValue = 0.001
run.Solver.Nonlinear.UseJacobian = True
run.Solver.Nonlinear.DerivativeEpsilon = 1e-16
run.Solver.Nonlinear.StepTol = 1e-30
run.Solver.Nonlinear.Globalization = 'LineSearch'
run.Solver.Linear.KrylovDimension = 70
run.Solver.Linear.MaxRestarts = 2

run.Solver.Linear.Preconditioner = 'PFMGOctree'
run.Solver.Linear.Preconditioner.PCMatrixType = 'FullJacobian'


# -----------------------------------------------------------------------------
# Run Simulation
# -----------------------------------------------------------------------------

run.run(working_directory=work_dir)
