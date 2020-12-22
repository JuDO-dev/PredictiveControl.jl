module PredictiveControl

################################################################################
# Main exports from this package
################################################################################
export ConstrainedTimeInvariantLQR,
       PrimalFullCondensing, PrimalNoCondensing, DualFullCondensing

# Analysis functions
export conditionBound

# Preconditioners
export circulantblockpreconditioner, sdpblockpreconditoner



################################################################################
# Packages we need
################################################################################
using BlockArrays
using ControlSystems
using ControlSystems: nstates, ninputs
using COSMO
using FFTW
using JuMP
using LinearAlgebra
using Logging
using MathOptInterface
using MatrixEquations
using OSQP

const MOI = MathOptInterface


# Overload these functions for our own types
import LinearAlgebra: cond

include( "utilities.jl" )

include( "types/problemTypes.jl" )
include( "types/clqr.jl" )


include( "matrices/fullCondensing.jl" )
include( "matrices/preconditioning.jl" )


include( "analysis/conditioning.jl")

# Solvers
include( "solvers/fastGradientMethod/fgm.jl" )
using .FastGradientMethod
const FGM = FastGradientMethod

export FGM, fastgradientmethod

# Solver interfaces
include( "solvers/interfaces/fgm_interface.jl" )
include( "solvers/interfaces/jump_interface.jl" )
include( "solvers/interfaces/osqp_interface.jl" )


end # module PredictiveControl
