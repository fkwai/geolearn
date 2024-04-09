from parflow import Run, write_pfb
import sys, os, parflow
import numpy as np
import shutil


seedLst = [1, 2, 3, 4, 5]
rainLst = [-1e-2, -5e-3,-1e-3, -4e-4,-1e-4, -5e-5]

for Seed in seedLst:
    for rain in rainLst:
        run_name = "ss_{}_{:.0e}".format(Seed, float(-rain))
        work_dir = os.path.join(r"/home/kuai/docker/parflow/", run_name)

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            
        os.mkdir(work_dir)

        run = Run(run_name)

        dx, dy, dz = 2, 1, 0.25
        nx, ny, nz = 50, 1, 40
        slopeX = 0.1
        leftP = 5

        # icp = np.full((nz, ny, nx), 0).astype(float)
        # for k in range(nx):
        #     icp[:, :, k] = leftP - slopeX * k * dx
        # write_pfb(os.path.join(work_dir, "icp.pfb"), icp)

        run.FileVersion = 4
        run.Process.Topology.P = 1
        run.Process.Topology.Q = 1
        run.Process.Topology.R = 1

        # ---------------------------------------------------------
        # Computational Grid
        # ---------------------------------------------------------
        run.ComputationalGrid.Lower.X = 0.0
        run.ComputationalGrid.Lower.Y = 0.0
        run.ComputationalGrid.Lower.Z = 0.0

        run.ComputationalGrid.DX = dx
        run.ComputationalGrid.DY = dy
        run.ComputationalGrid.DZ = dz

        run.ComputationalGrid.NX = nx
        run.ComputationalGrid.NY = ny
        run.ComputationalGrid.NZ = nz

        # ---------------------------------------------------------
        # Domain Geometry
        # ---------------------------------------------------------
        run.GeomInput.Names = "boxinput"

        run.GeomInput.boxinput.InputType = "Box"
        run.GeomInput.boxinput.GeomName = "domain"

        # -----------------------------------------------------------------------------
        # Domain Geometry
        # -----------------------------------------------------------------------------
        run.Geom.domain.Lower.X = 0.0
        run.Geom.domain.Lower.Y = 0.0
        run.Geom.domain.Lower.Z = 0.0

        run.Geom.domain.Upper.X = nx * dx
        run.Geom.domain.Upper.Y = ny * dy
        run.Geom.domain.Upper.Z = nz * dz

        run.Geom.domain.Patches = "x_lower x_upper y_lower y_upper z_lower z_upper"

        # -----------------------------------------------------------------------------
        # Perm
        # -----------------------------------------------------------------------------

        run.Geom.Perm.Names = "domain"

        # Values in m/hour

        # these are examples to make the upper portions of the v heterogeneous
        # the following is ignored if the perm.type 'Constant' settings are not
        # commented out, below.

        run.Geom.domain.Perm.Type = "TurnBands"
        run.Geom.domain.Perm.LambdaX = 50.0
        run.Geom.domain.Perm.LambdaY = 50.0
        run.Geom.domain.Perm.LambdaZ = 0.5
        run.Geom.domain.Perm.GeomMean = 1.0

        run.Geom.domain.Perm.Sigma = 1.0
        run.Geom.domain.Perm.NumLines = 40
        run.Geom.domain.Perm.RZeta = 5.0
        run.Geom.domain.Perm.KMax = 100.0
        run.Geom.domain.Perm.DelK = 0.2
        run.Geom.domain.Perm.Seed = Seed
        run.Geom.domain.Perm.LogNormal = "Log"
        run.Geom.domain.Perm.StratType = "Bottom"

        run.Perm.TensorType = "TensorByGeom"
        run.Geom.Perm.TensorByGeom.Names = "domain"
        run.Geom.domain.Perm.TensorValX = 1.0
        run.Geom.domain.Perm.TensorValY = 1.0
        run.Geom.domain.Perm.TensorValZ = 1.0

        # -----------------------------------------------------------------------------
        # Specific Storage
        # -----------------------------------------------------------------------------

        run.SpecificStorage.Type = "Constant"
        run.SpecificStorage.GeomNames = "domain"
        run.Geom.domain.SpecificStorage.Value = 1.0e-5

        # -----------------------------------------------------------------------------
        # Phases
        # -----------------------------------------------------------------------------

        run.Phase.Names = "water"

        run.Phase.water.Density.Type = "Constant"
        run.Phase.water.Density.Value = 1.0

        run.Phase.water.Viscosity.Type = "Constant"
        run.Phase.water.Viscosity.Value = 1.0

        # -----------------------------------------------------------------------------
        # Contaminants
        # -----------------------------------------------------------------------------

        run.Contaminants.Names = ""

        # -----------------------------------------------------------------------------
        # Retardation
        # -----------------------------------------------------------------------------

        run.Geom.Retardation.GeomNames = ""

        # -----------------------------------------------------------------------------
        # Gravity
        # -----------------------------------------------------------------------------

        run.Gravity = 1.0

        # -----------------------------------------------------------------------------
        # Setup timing info
        # -----------------------------------------------------------------------------

        # run for 2 hours @ 6min timesteps
        #
        run.TimingInfo.BaseUnit = 1.0
        run.TimingInfo.StartCount = 0
        run.TimingInfo.StartTime = 0.0
        run.TimingInfo.StopTime = 120000.0
        run.TimingInfo.DumpInterval = -1
        run.TimeStep.Type = "Constant"
        run.TimeStep.Value = 500.0
        #
        # -----------------------------------------------------------------------------
        # Porosity
        # -----------------------------------------------------------------------------

        run.Geom.Porosity.GeomNames = "domain"


        run.Geom.domain.Porosity.Type = "Constant"
        run.Geom.domain.Porosity.Value = 0.1

        # -----------------------------------------------------------------------------
        # Domain
        # -----------------------------------------------------------------------------

        run.Domain.GeomName = "domain"

        # -----------------------------------------------------------------------------
        # Relative Permeability
        # -----------------------------------------------------------------------------

        run.Phase.RelPerm.Type = "VanGenuchten"
        run.Phase.RelPerm.GeomNames = "domain"

        run.Geom.domain.RelPerm.Alpha = 6.0
        run.Geom.domain.RelPerm.N = 2.0

        # ---------------------------------------------------------
        # Saturation
        # ---------------------------------------------------------

        run.Phase.Saturation.Type = "VanGenuchten"
        run.Phase.Saturation.GeomNames = "domain"

        run.Geom.domain.Saturation.Alpha = 6.0
        run.Geom.domain.Saturation.N = 2.0
        run.Geom.domain.Saturation.SRes = 0.2
        run.Geom.domain.Saturation.SSat = 1.0


        # -----------------------------------------------------------------------------
        # Wells
        # -----------------------------------------------------------------------------
        run.Wells.Names = ""

        # -----------------------------------------------------------------------------
        # Time Cycles
        # -----------------------------------------------------------------------------
        run.Cycle.Names = "constant"
        run.Cycle.constant.Names = "alltime"
        run.Cycle.constant.alltime.Length = 1
        run.Cycle.constant.Repeat = -1

        #
        # -----------------------------------------------------------------------------
        # Boundary Conditions: Pressure
        # -----------------------------------------------------------------------------
        run.BCPressure.PatchNames = run.Geom.domain.Patches

        # run.Patch.x_lower.BCPressure.Type = 'FluxConst'
        # run.Patch.x_lower.BCPressure.Cycle = 'constant'
        # run.Patch.x_lower.BCPressure.alltime.Value = 0.0
        run.Patch.x_lower.BCPressure.Type = "DirEquilRefPatch"
        run.Patch.x_lower.BCPressure.Cycle = "constant"
        run.Patch.x_lower.BCPressure.RefGeom = "domain"
        run.Patch.x_lower.BCPressure.RefPatch = "z_lower"
        run.Patch.x_lower.BCPressure.alltime.Value = leftP

        run.Patch.y_lower.BCPressure.Type = "FluxConst"
        run.Patch.y_lower.BCPressure.Cycle = "constant"
        run.Patch.y_lower.BCPressure.alltime.Value = 0.0

        run.Patch.z_lower.BCPressure.Type = "FluxConst"
        run.Patch.z_lower.BCPressure.Cycle = "constant"
        run.Patch.z_lower.BCPressure.alltime.Value = 0.0

        run.Patch.x_upper.BCPressure.Type = "FluxConst"
        run.Patch.x_upper.BCPressure.Cycle = "constant"
        run.Patch.x_upper.BCPressure.alltime.Value = 0.0

        run.Patch.y_upper.BCPressure.Type = "FluxConst"
        run.Patch.y_upper.BCPressure.Cycle = "constant"
        run.Patch.y_upper.BCPressure.alltime.Value = 0.0

        ## overland flow boundary condition with very heavy rainfall then slight ET
        run.Patch.z_upper.BCPressure.Type = "OverlandFlow"
        run.Patch.z_upper.BCPressure.Cycle = "constant"
        run.Patch.z_upper.BCPressure.alltime.Value = rain

        # ---------------------------------------------------------
        # Topo slopes in x-direction
        # ---------------------------------------------------------

        run.TopoSlopesX.Type = "Constant"
        run.TopoSlopesX.GeomNames = "domain"
        run.TopoSlopesX.Geom.domain.Value = slopeX

        # ---------------------------------------------------------
        # Topo slopes in y-direction
        # ---------------------------------------------------------


        run.TopoSlopesY.Type = "Constant"
        run.TopoSlopesY.GeomNames = "domain"
        run.TopoSlopesY.Geom.domain.Value = 0.00

        # ---------------------------------------------------------
        # Mannings coefficient
        # ---------------------------------------------------------

        run.Mannings.Type = "Constant"
        run.Mannings.GeomNames = "domain"
        run.Mannings.Geom.domain.Value = 1.0e-6
        # run.Mannings.Geom.domain.Value = 1e-3


        # -----------------------------------------------------------------------------
        # Phase sources:
        # -----------------------------------------------------------------------------

        run.PhaseSources.water.Type = "Constant"
        run.PhaseSources.water.GeomNames = "domain"
        run.PhaseSources.water.Geom.domain.Value = 0.0

        # -----------------------------------------------------------------------------
        # Exact solution specification for error calculations
        # -----------------------------------------------------------------------------

        run.KnownSolution = "NoKnownSolution"


        # -----------------------------------------------------------------------------
        # Set solver parameters
        # -----------------------------------------------------------------------------

        run.Solver = "Richards"
        run.Solver.TerrainFollowingGrid = True

        run.Solver.MaxIter = 2500

        run.Solver.Nonlinear.MaxIter = 300
        run.Solver.Nonlinear.ResidualTol = 1e-6
        run.Solver.Nonlinear.EtaChoice = "Walker1"
        run.Solver.Nonlinear.EtaValue = 0.001
        run.Solver.Nonlinear.UseJacobian = True
        run.Solver.Nonlinear.DerivativeEpsilon = 1e-16
        run.Solver.Nonlinear.StepTol = 1e-20
        run.Solver.Nonlinear.Globalization = "LineSearch"
        run.Solver.Linear.KrylovDimension = 20
        run.Solver.Linear.MaxRestart = 2

        run.Solver.Linear.Preconditioner = "MGSemi"
        run.Solver.Linear.Preconditioner = "PFMG"

        run.Solver.Linear.Preconditioner.MGSemi.MaxIter = 1
        run.Solver.Linear.Preconditioner.MGSemi.MaxLevels = 10
        run.Solver.PrintSubsurf = False
        run.Solver.Drop = 1e-20
        run.Solver.AbsTol = 1e-12
        #
        run.Solver.WriteSiloSubsurfData = True
        run.Solver.WriteSiloPressure = True
        run.Solver.WriteSiloSaturation = True

        run.Solver.WriteSiloSlopes = True
        run.Solver.PrintSlopes = True
        run.Solver.WriteSiloMask = True
        run.Solver.WriteSiloEvapTrans = True
        run.Solver.WriteSiloEvapTransSum = True
        run.Solver.WriteSiloOverlandSum = True
        run.Solver.WriteSiloMannings = True
        run.Solver.WriteSiloSpecificStorage = True

        # ---------------------------------------------------------
        # Initial conditions: water pressure
        # ---------------------------------------------------------

        # set water table to be at the bottom of the domain, the top layer is initially dry
        run.ICPressure.Type = "HydroStaticPatch"
        run.ICPressure.GeomNames = "domain"
        run.Geom.domain.ICPressure.Value = 0.0
        run.Geom.domain.ICPressure.RefGeom = "domain"
        run.Geom.domain.ICPressure.RefPatch = "z_lower"

        # run.ICPressure.Type = "PFBFile"
        # run.ICPressure.GeomNames = "domain"
        # run.Geom.domain.ICPressure.FileName = "icp.pfb"
        # run.Geom.domain.ICPressure.RefGeom = "domain"
        # run.Geom.domain.ICPressure.RefPatch = "bottom"


        # -----------------------------------------------------------------------------
        # Run and Unload the ParFlow output files
        # -----------------------------------------------------------------------------

        run.run(working_directory=work_dir)
