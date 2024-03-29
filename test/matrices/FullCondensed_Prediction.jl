using BlockArrays
using ControlSystems
using LinearAlgebra
using PredictiveControl

const MPC = PredictiveControl

# Some utilities for testing
include( "../../src/utilities.jl" )
include( "../testUtils.jl" )


# Create a sample system
A = [1.0 0.0 0.0 0.0;
     0.0 2.0 0.0 0.0;
     0.0 0.0 3.0 0.0;
     0.0 0.0 0.0 4.0];
B = [1.0 0.0;
     0.0 0.0;
     0.0 1.0;
     0.0 0.0];
C = to_matrix( Float64, I, 4 )

sys = StateSpace( A, B, C, 0, 0.1 )

N  = 5
nx = ControlSystems.nstates( sys )
nu = ControlSystems.ninputs( sys )

###################################################################
# Test the propagation of the input through the system
###################################################################

# Horizon must be greater than 1
@test_throws DomainError MPC.prediction( sys, -1 )
@test_throws DomainError MPC.prediction( sys,  0 )

# Form a prediction matrix to test
mat = MPC.prediction( sys, N )

@test blocksize( mat ) == (N, N)
@test size( mat ) == (nx*N, nu*N)

# Make sure the blocks have the proper values for the system
for i = 1:N, j = i:N
    b = view( mat, Block( j, i ) )

    @test size( b ) == (nx, nu)
    @test b == A^(j-i) * B
end
