using ControlSystems
using PredictiveControl
using Test

include( "../testUtils.jl" )

clqrUnstable  = createSampleLTISystem()
clqrStablized = createSampleLTISystem( usek = true )
clqrWithS     = createSampleLTISystem( uses = true, usek = true )

# These are all conditions that are not allowed
@test_throws DomainError conditionBound( clqrUnstable )
@test_throws DomainError conditionBound( clqrWithS )

# A test for a single-input pre-stabilized system
@test conditionBound( clqrStablized ) ≈ 9.986301 atol=1e-4


# Setup a test for a multi-input system (it means actual matrices are used in the computation)
A = [0.9 1.0;
     0.0 0.9]
B = [0.0 1.0;
     1.0 0.0]
C = [1.0 0.0;
     0.0 1.0]
D = [0.0 0.0;
     0.0 0.0]
Ts = 0.1
sys = StateSpace( A, B, C, D, Ts )

Q = [1.0 1.0;
     1.0 1.0]
R = [1.0 0.0
     0.0 1.0]

clqrTwoInputs = ConstrainedTimeInvariantLQR( sys, 5, Q, R, :Q )

@test conditionBound( clqrTwoInputs ) ≈ 12201.000000000648
