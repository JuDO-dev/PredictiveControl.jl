using BlockArrays
using ControlSystems
using LinearAlgebra
using PredictiveControl
using Test

function testconstraint( G, L, g, u, x₀, expec )
    res = G*u - g - L*x₀
    @test res ≈ expec atol=1e-4
end

n = 2
m = 1

# This is a controllable and stable discrete-time system
Astab = [0.9 1.0
         0.0 0.9]

B = [0.0;
     1.0]
C = [1.0 0.0;
     0.0 1.0]
D = [0.0;
     0.0]
Ts = 0.1

sysStab = StateSpace( Astab, B, C, D, Ts )

Q = [1.0 1.0;
     1.0 1.0]
R = [1.0]
N = 2

###############################################
## Test only input constraints
###############################################
E = [ 0.0 0.0;
      0.0 0.0 ]
F = [  1.0;
      -1.0 ]
g = [ 5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

# Test sizes
@test size(ḡ) == (2*N,)
@test size(L) == (2*N, n)
@test size(G) == (2*N, m*N)

@test reshape( G[Block(1,1)], (:) ) == F
@test reshape( G[Block(2,2)], (:) ) == F
@test G[Block(1,2)] == zeros( Float64, (2, m) )
@test G[Block(2,1)] == zeros( Float64, (2, m) )

@test L == zeros( Float64, (2*N, n) )

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g

# Test with all inputs in bounds
u  = [ 1.0;
      -1.0 ]
x₀ = [ 1.0;
       1.0 ]

e  = [    u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with all inputs out of bounds
u = [ 6.0;
     -6.0 ]
x₀ = [ 1.0;
       1.0 ]


e  = [    u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with one inbounds, one out of bounds
u = [ 1.0;
     -6.0 ]
x₀ = [ 1.0;
       1.0 ]

e  = [    u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )


###############################################
## Test only state constraints
###############################################
E = [ 1.0 0.0;
      0.0 1.0 ]
F = [ 0.0;
      0.0 ]
g = [ 5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

sys = PredictiveControl.getsystem( clqr )

# Test sizes
@test size(ḡ) == (2*N,)
@test size(L) == (2*N, n)
@test size(G) == (2*N, m*N)

@test G == PredictiveControl.prediction( sys, clqr.N )

@test L[Block(1,1)] == -E*sys.A
@test L[Block(2,1)] == -E*sys.A^2

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g

# Test with all states in bounds
u  = [ 0.0;
       0.0 ]
x₀ = [ 0.0;
       0.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [ x₁[1] - g[1];    # States @t=1
       x₁[2] - g[2];    # States @t=1
       x₂[1] - g[1];    # States @t=2
       x₂[2] - g[2] ]   # States @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with all states out of bounds
u  = [ 6.0;
       6.0 ]
x₀ = [ 20.0;
       20.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [ x₁[1] - g[1];    # States @t=1
       x₁[2] - g[2];    # States @t=1
       x₂[1] - g[1];    # States @t=2
       x₂[2] - g[2] ]   # States @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with one inbounds, one out of bounds
u  = [  5.0;
        5.0 ]
x₀ = [ 0.0;
       0.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [ x₁[1] - g[1];    # States @t=1
       x₁[2] - g[2];    # States @t=1
       x₂[1] - g[1];    # States @t=2
       x₂[2] - g[2] ]   # States @t=2

testconstraint( G, L, ḡ, u, x₀, e )


###############################################
## Test the inclusion of both state and input constraints
###############################################
E = [ 1.0 0.0;
      0.0 1.0;
      0.0 0.0;
      0.0 0.0 ]
F = [ 0.0;
      0.0;
      1.0;
     -1.0 ]
g = [ 5.0;
      5.0;
      5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

sys = PredictiveControl.getsystem( clqr )

# Test sizes
@test size(ḡ) == (4*N,)
@test size(L) == (4*N, n)
@test size(G) == (4*N, m*N)

@test G[Block(1,1)] == E*sys.B + F
@test G[Block(2,2)] == E*sys.B + F
@test G[Block(2,1)] == E*sys.A*sys.B
@test G[Block(1,2)] == zeros( Float64, (4, m) )

@test L[Block(1,1)] == -E*sys.A
@test L[Block(2,1)] == -E*sys.A^2

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g

# Test with all states and inputs in bounds
u  = [ 0.0;
       0.0 ]
x₀ = [ 0.0;
       0.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [   x₁[1] - g[1];      # States @t=1
         x₁[2] - g[2];      # States @t=1
          u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
         x₂[1] - g[1];      # States @t=2
         x₂[2] - g[2];      # States @t=2
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with all inputs out of bounds, states out of bounds
u  = [ 6.0;
       6.0 ]
x₀ = [ 20.0;
       20.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [   x₁[1] - g[1];      # States @t=1
         x₁[2] - g[2];      # States @t=1
          u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
         x₂[1] - g[1];      # States @t=2
         x₂[2] - g[2];      # States @t=2
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )

# Test with one inbounds, one out of bounds
u  = [ 5.0;
       6.0 ]
x₀ = [ 0.0;
       0.0 ]

x₁ = sysStab.A*x₀ + sysStab.B*u[1]
x₂ = sysStab.A*x₁ + sysStab.B*u[2]

e  = [   x₁[1] - g[1];      # States @t=1
         x₁[2] - g[2];      # States @t=1
          u[1] - g[1];      # Inputs @t=1
       -( u[1] + g[2] );    # Inputs @t=1
         x₂[1] - g[1];      # States @t=2
         x₂[2] - g[2];      # States @t=2
          u[2] - g[1];      # Inputs @t=2
       -( u[2] + g[2] ) ]   # Inputs @t=2

testconstraint( G, L, ḡ, u, x₀, e )


###############################################
## Test the inclusion of input constraints with a pre-stabilized system
###############################################
E = [ 0.0 0.0;
      0.0 0.0 ]
F = [ 1.0;
     -1.0 ]
g = [ 5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, K = :dlqr, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

sys = PredictiveControl.getsystem( clqr )

# Test sizes
@test size(ḡ) == (2*N,)
@test size(L) == (2*N, n)
@test size(G) == (2*N, m*N)

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g


# Test with all inputs in bounds
v  = [ 0.0;
       0.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    u₀ - g[1];      # Inputs @t=0
       -( u₀ + g[2] );    # Inputs @t=0
          u₁ - g[1];      # Inputs @t=1
       -( u₁ + g[2] ) ]   # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )

# Test with all inputs out of bounds
v  = [ 40.0;
       30.0 ]
x₀ = [ 20.0;
       20.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    u₀ - g[1];      # Inputs @t=0
       -( u₀ + g[2] );    # Inputs @t=0
          u₁ - g[1];      # Inputs @t=1
       -( u₁ + g[2] ) ]   # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )

# Test with one inbounds, one out of bounds
v  = [ 5.0;
       10.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    u₀ - g[1];      # Inputs @t=0
       -( u₀ + g[2] );    # Inputs @t=0
          u₁ - g[1];      # Inputs @t=1
       -( u₁ + g[2] ) ]   # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )


###############################################
## Test the inclusion of state constraints with a pre-stabilized system
###############################################
E = [ 1.0 0.0;
      0.0 1.0 ]
F = [ 0.0;
      0.0 ]
g = [ 5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, K = :dlqr, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

sys = PredictiveControl.getsystem( clqr )

# Test sizes
@test size(ḡ) == (2*N,)
@test size(L) == (2*N, n)
@test size(G) == (2*N, m*N)

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g


# Test with all states in bounds
v  = [ 0.0;
       0.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [ x₁[1] - g[1];    # State @t=1
       x₁[2] - g[2];    # State @t=1
       x₂[1] - g[1];    # State @t=2
       x₂[2] - g[2] ]   # State @t=2

testconstraint( G, L, ḡ, v, x₀, e )

# Test with all states out of bounds
v  = [ 30.0;
       30.0 ]
x₀ = [ 20.0;
       20.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [ x₁[1] - g[1];    # State @t=1
       x₁[2] - g[2];    # State @t=1
       x₂[1] - g[1];    # State @t=2
       x₂[2] - g[2] ]   # State @t=2

testconstraint( G, L, ḡ, v, x₀, e )

# Test with one inbounds, one out of bounds
v  = [ 5.0;
       10.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [ x₁[1] - g[1];    # State @t=1
       x₁[2] - g[2];    # State @t=1
       x₂[1] - g[1];    # State @t=2
       x₂[2] - g[2] ]   # State @t=2

testconstraint( G, L, ḡ, v, x₀, e )


###############################################
## Test the inclusion of both state and input constraints on a pre-stabilized system
###############################################
E = [ 1.0 0.0;
      0.0 1.0;
      0.0 0.0;
      0.0 0.0 ]
F = [ 0.0;
      0.0;
      1.0;
     -1.0 ]
g = [ 5.0;
      5.0;
      5.0;
      5.0 ]

clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, K = :dlqr, F = F , E = E, g = g )
G, L, ḡ = PredictiveControl.inequalityconstraints( clqr, PrimalFullCondensing )

sys = PredictiveControl.getsystem( clqr )

# Test sizes
@test size(ḡ) == (4*N,)
@test size(L) == (4*N, n)
@test size(G) == (4*N, m*N)

@test ḡ[Block(1)] == g
@test ḡ[Block(2)] == g

# Test with all states and inputs in bounds
v  = [ 0.0;
       0.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    x₁[1] - g[1];     # State  @t=1
          x₁[2] - g[2];     # State  @t=1
          u₀    - g[3];     # Inputs @t=0
       -( u₀    + g[4] );   # Inputs @t=0
          x₂[1] - g[1];     # State  @t=2
          x₂[2] - g[2]      # State  @t=2
          u₁    - g[3];     # Inputs @t=1
       -( u₁    + g[4] ) ]  # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )

# Test with all inputs out of bounds, states out of bounds
v  = [ 6.0;
       6.0 ]
x₀ = [ 20.0;
       20.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    x₁[1] - g[1];     # State  @t=1
          x₁[2] - g[2];     # State  @t=1
          u₀    - g[3];     # Inputs @t=0
       -( u₀    + g[4] );   # Inputs @t=0
          x₂[1] - g[1];     # State  @t=2
          x₂[2] - g[2]      # State  @t=2
          u₁    - g[3];     # Inputs @t=1
       -( u₁    + g[4] ) ]  # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )

# Test with one inbounds, one out of bounds
v  = [ 5.0;
       6.0 ]
x₀ = [ 0.0;
       0.0 ]

u₀ = -(clqr.K*x₀)[1] + v[1]
x₁ = sysStab.A*x₀ + sysStab.B*u₀
u₁ = -(clqr.K*x₁)[1] + v[2]
x₂ = sysStab.A*x₁ + sysStab.B*u₁

e  = [    x₁[1] - g[1];     # State  @t=1
          x₁[2] - g[2];     # State  @t=1
          u₀    - g[3];     # Inputs @t=0
       -( u₀    + g[4] );   # Inputs @t=0
          x₂[1] - g[1];     # State  @t=2
          x₂[2] - g[2]      # State  @t=2
          u₁    - g[3];     # Inputs @t=1
       -( u₁    + g[4] ) ]  # Inputs @t=1

testconstraint( G, L, ḡ, v, x₀, e )
