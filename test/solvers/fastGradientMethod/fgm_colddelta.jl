using BlockArrays
using ControlSystems
using LinearAlgebra
using PredictiveControl
using Test

# Setup the problem using a test case from Richter's paper
n = 4
m = 2
N = 5

A = [ 0.7 -0.1  0.0  0.0;
      0.2 -0.5  0.1  0.0;
      0.0  0.1  0.1  0.0;
      0.5  0.0  0.5  0.5 ]

B = [ 0.0  0.1;
      0.1  1.0;
      0.1  0.0;
      0.0  0.0 ]

C = Matrix{Float64}( I, 4, 4 )

D = fill( 0.0, (4, 2) )

sys = StateSpace( A, B, C, D, 0.01 )

Gₛ = [ 1.0  0.0;
      -1.0  0.0;
       0.0  1.0;
       0.0 -1.0 ]
gₛ = [ 1.0;
      -1.0;
       1.0;
      -1.0 ]

clqr = ConstrainedTimeInvariantLQR( sys, N, Matrix( 1.0*I(n) ), Matrix( 1.0*I(m) ), :Q )

H = PredictiveControl.hessian( clqr, PrimalFullCondensing )
G = kron( I(N), Gₛ )
g = reshape( kron( ones( Float64, N, 1 ), gₛ ), (:) )

@test isapprox( FGM.coldstartdelta( G, g, H = Matrix( H ) ), 19.026675, atol = 1e-4 )
@test isapprox( FGM.coldstartdelta( G, g, L = eigmax( Matrix( H ) ) ), 19.026675, atol = 1e-4 )
