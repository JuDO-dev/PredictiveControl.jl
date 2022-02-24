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
     0.0 0.0 0.0 4.0]
B = [1.0 0.0;
     0.0 0.0;
     0.0 1.0;
     0.0 0.0]
C = to_matrix( Float64, I, 4 )

sys = StateSpace( A, B, C, 0, 0.1 )

N  = 5
nx = ControlSystems.nstates( sys )
nu = ControlSystems.ninputs( sys )

###################################################################
# Test the propagation of the input through the system
###################################################################

# Horizon must be greater than 1
@test_throws DomainError MPC.initialpropagation( sys, -1 )
@test_throws DomainError MPC.initialpropagation( sys,  0 )

# Form a propagation matrix to test
mat = MPC.initialpropagation( sys, N )

@test blocksize( mat ) == (N, 1)
@test size( mat ) == (nx*N, nx)

# Make sure the blocks have the proper values for the system
for i = 1:N
    b = view( mat, Block( i, 1 ) )

    @test size( b ) == (nx, nx)
    @test b == sys.A^i
end

# Form a propagation matrix to test
mat = MPC.initialpropagation( sys, N, ive = true )

@test blocksize( mat ) == (N, 1)
@test size( mat ) == (nx*N, nx)

# Make sure the blocks have the proper values for the system
for i = 0:N-1
    b = view( mat, Block( i+1, 1 ) )

    @test size( b ) == (nx, nx)
    @test b == sys.A^i
end
