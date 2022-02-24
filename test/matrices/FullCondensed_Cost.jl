using BlockArrays
using ControlSystems
using LinearAlgebra
using PredictiveControl
using Test

const MPC = PredictiveControl

# Some utilities for testing
include( "../../src/utilities.jl" )
include( "../testUtils.jl" )


# Create a sample system
A = [1.0 0.0 0.0 0.0;
     1.0 2.0 0.0 0.0;
     0.0 0.0 3.0 0.0;
     0.0 0.0 1.0 4.0]
B = [1.0 0.0;
     0.0 0.0;
     0.0 1.0;
     0.0 0.0]
C = to_matrix( Float64, I, 4 )

sys = StateSpace( A, B, C, 0, 0.1 )

Q = diagm( [1.0, 2.0, 3.0, 4.0] )
R = diagm( [1.0, 2.0] )

N  = 5
nx = ControlSystems.nstates( sys )
nu = ControlSystems.ninputs( sys )


###################################################################
# Test the matrices of an uncontrolled system
###################################################################
clqr = ConstrainedTimeInvariantLQR( sys, N, Q, R, :Q )

# Form a propagation matrix to test
H = MPC.hessian( clqr, PrimalFullCondensing )
J = MPC.linearcoefficients( clqr, PrimalFullCondensing )

@test blocksize( H ) == (N, N)
@test size( H ) == (nu*N, nu*N)

@test blocksize( J ) == (N, 1)
@test size( J ) == (nu*N, nx)

# Make sure the blocks have the proper values for the system
for i = 1:N
#    b = view( mat, Block( i, 1 ) )

#    @test size( b ) == (nx, nx)
#    @test b == sys.A^i
end


###################################################################
# Test the matrices of a controlled system
###################################################################
clqr = ConstrainedTimeInvariantLQR( sys, N, Q, R, :Q, K = :dlqr )

# Form a propagation matrix to test
H = MPC.hessian( clqr, PrimalFullCondensing )
J = MPC.linearcoefficients( clqr, PrimalFullCondensing )

@test blocksize( H ) == (N, N)
@test size( H ) == (nu*N, nu*N)

@test blocksize( J ) == (N, 1)
@test size( J ) == (nu*N, nx)

# Make sure the blocks have the proper values for the system
for i = 1:N
#    b = view( mat, Block( i, 1 ) )

#    @test size( b ) == (nx, nx)
#    @test b == sys.A^i
end
