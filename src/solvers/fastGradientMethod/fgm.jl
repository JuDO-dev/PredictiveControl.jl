module FastGradientMethod

using DocStringExtensions
using LinearAlgebra
using OSQP
using Printf
using Roots
using SparseArrays

using Base.Iterators

import Base: iterate

export fastgradientmethod

# Include the various iteration helpers
include( "../iterationUtils/stopping.jl" )
include( "../iterationUtils/loop.jl" )
include( "../iterationUtils/apply.jl" )

"""
```julia
    AbstractStep
```

An abstract super type that is used to specify the step length computation in the
Fast Gradient Method. Implementations of the step length computation should define
a type that inherits from this abstract type.
"""
abstract type AbstractStep end


# The actual FGM algorithm
include( "./fgm_algorithm.jl" )
include( "./fgm_stopping.jl" )
include( "./fgm_stepping.jl")
include( "./fgm_analysis.jl" )

end
