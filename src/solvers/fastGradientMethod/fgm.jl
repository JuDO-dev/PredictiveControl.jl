module FastGradientMethod

using LinearAlgebra
using OSQP
using SparseArrays
using Roots

export fastgradientmethod

include( "./fgm_algorithm.jl" )
include( "./fgm_analysis.jl" )

end
