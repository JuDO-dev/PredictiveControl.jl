using ControlSystems
using PredictiveControl
using Test

# This is a controllable and unstable discrete-time system
Aunstab = [2.0 1.0;
           0.0 1.0]

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

sysStab   = StateSpace( Astab, B, C, D, Ts )
sysUnstab = StateSpace( Aunstab, B, C, D, Ts )

Q = [1.0 1.0;
     1.0 1.0]
R = [1.0]
N = 2

em = Matrix{Float64}(undef, 0, 0)

# Ensure the dimension checks on the weight matrices work
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, em,  R,  Q )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N,  Q, em,  Q )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N,  Q,  R, em )

# Define an incorrect controller
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, K = [ 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, K = [ 1.0; 2.0 ] )

# Define bad constraints
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 ],     E = [ 1.0 ],      g = [ 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 2.0 ], E = [ 1.0 1.0 ],  g = [ 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 2.0 ], E = [ 1.0 ],      g = [ 1.0; 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 2.0 ], E = [ 1.0; 1.0 ], g = [ 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 2.0 ], E = [ 1.0; 1.0 ], g = [ 1.0 ] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, F = [ 1.0 2.0; 1.0 2.0], E = [ 1.0 ], g = [ 1.0 ] )

# Define bad bounds
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xₗ = [1.0; 2.0; 3.0] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xᵤ = [1.0; 2.0; 3.0] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, uₗ = [1.0; 2.0] )
@test_throws DimensionMismatch ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, uᵤ = [1.0; 2.0] )

# Test bounds constraints
let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xᵤ=1.0 )
    @test length( clqr.xᵤ ) == 2
    @test clqr.xᵤ[1] == 1.0
    @test clqr.xᵤ[2] == 1.0
end

let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xₗ=1.0 )
    @test length( clqr.xₗ ) == 2
    @test clqr.xₗ[1] == 1.0
    @test clqr.xₗ[2] == 1.0
end

let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, uᵤ=1.0 )
    @test length( clqr.uᵤ ) == 1
    @test clqr.uᵤ[1] == 1.0
end

let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, uₗ=1.0 )
    @test length( clqr.uₗ ) == 1
    @test clqr.uₗ[1] == 1.0
end

let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xᵤ=[Inf; 1.0] )
    @test length( clqr.xᵤ ) == 2
    @test ismissing( clqr.xᵤ[1] )
    @test clqr.xᵤ[2] == 1.0
end

let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :Q, xₗ=[-Inf; 1.0] )
    @test length( clqr.xₗ ) == 2
    @test ismissing( clqr.xₗ[1] )
    @test clqr.xₗ[2] == 1.0
end

# Use P=Q and no prestabilization controller
let clqr = ConstrainedTimeInvariantLQR( sysUnstab, N, Q, R, :Q )
    @test iszero( clqr.K )

    @test clqr.Qₖ == clqr.Q
    @test clqr.Q  == clqr.P

    @test !isstable( PredictiveControl.getsystem( clqr, prestabilized = false ) )
    @test !isstable( PredictiveControl.getsystem( clqr ) )
end

# Use P=Q and an LQR prestabilization controller
let clqr = ConstrainedTimeInvariantLQR( sysUnstab, N, Q, R, :Q, K = :dlqr )
    @test clqr.Qₖ != clqr.Q
    @test clqr.Qₖ == clqr.P

    @test !isstable( PredictiveControl.getsystem( clqr, prestabilized = false ) )
    @test isstable( PredictiveControl.getsystem( clqr ) )
end

# Use P=(the solution to the DARE) and the LQR controller for prestabilization
let clqr = ConstrainedTimeInvariantLQR( sysUnstab, N, Q, R, :dare, K = :dlqr )
    Aₖ = Aunstab - B*clqr.K

    @test clqr.Qₖ != clqr.Q
    @test clqr.Qₖ != clqr.P
    @test clqr.P  ≈ Aₖ'*clqr.P*Aₖ + clqr.Qₖ

    @test !isstable( PredictiveControl.getsystem( clqr, prestabilized = false ) )
    @test isstable( PredictiveControl.getsystem( clqr ) )
end

# Use P=(the solution to the lyap) and no stabilization
let clqr = ConstrainedTimeInvariantLQR( sysStab, N, Q, R, :dlyap )
    @test iszero( clqr.K )

    @test clqr.Qₖ == clqr.Q
    @test clqr.Q  != clqr.P
    @test clqr.P  ≈ Astab'*clqr.P*Astab + clqr.Q
end
