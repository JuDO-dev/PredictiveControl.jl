"""
    prediction( sys::StateSpace, N::Integer )

Form the prediction matrix ``Γ`` that computes the states of the linear system `sys` over the horizon of length `N`
by using all the inputs across the entire horizon.


# Extended help

This matrix consists of the blocks ``A^i B`` where ``i`` is the diagonal number, forming the matrix
```math
⎡ B   •   •  •  ⋯⎤
  ⎢ AB  B   •  •  ⋯⎥
  ⎢ A²B AB  B  •  ⋯⎥
  ⎢ A³B A²B AB B  ⋯⎥
  ⎣ ⋮   ⋮   ⋮  ⋮ ⋱ ⎦
```

It is used to compute the component of the state vector for a horizon that is caused by the system inputs
during the horizon using the equation ``x = Γu``.

The full state vector for a horizon can be computed as ``x = Γu + Φx₀``, where ``Φ`` is the initial condition
propagation matrix from [`initialpropagation`](@ref).
"""
function prediction( sys::StateSpace, N::Integer )
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    nu = ninputs( sys )
    nx = nstates( sys )
    mt = eltype( sys.A )

    mainDiag  = kron( I(N), sys.B )

    otherDiag = kron( I(N), to_matrix( mt, I, nx ) )

    D = zeros( mt, N, N )
    D[diagind( D, -1 )] .= 1

    otherDiag = otherDiag + kron( D, -sys.A )

    Γ = otherDiag\mainDiag

    return BlockArray{ mt }( Γ, [nx for i = 1:N], [nu for i = 1:N] )
end


"""
    initialpropagation( sys::StateSpace, N::Integer )

Form the matrix ``Φ`` that propagates the initial condition ``x₀`` in the linear system `sys` across the horizon `N`.

# Extended help

This matrix consists of a column of the blocks ``A^i`` where ``i`` is the row number, forming the matrix
```math
⎡ A  ⎤
  ⎢ A² ⎥
  ⎢ A³ ⎥
  ⎢ A⁴ ⎥
  ⎣ ⋮  ⎦
```

It is used to compute the component of the state vector for a horizon that is caused by the intial state of the system using
the equation ``x = Φx₀``.

The full state vector for a horizon can be computed as ``x = Γu + Φx₀``, where ``Γ`` is the prediction matrix
from [`prediction`](@ref).
"""
function initialpropagation( sys::StateSpace, N::Integer )
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    blockGen = (sys.A^i for i=1:N)

    # The mortar function requires the passed array to have ndims=2, so reshape to
    # have an array with 2 dimensions even though there is only one column
    blocks = reshape( collect( blockGen ),  (N, 1) )

    return mortar( blocks )
end


function inputprediction( sys::StateSpace, K::AbstractArray, N::Integer )
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    nu = ninputs( sys )
    nx = nstates( sys )
    mt = eltype( sys.A )

    # Create the prediction matrix for the controlled system
    Γ = prediction( sys, N )
    K̄ = blockkron( 1.0*I(N), K )

    zerorow = zeros( Float64, (nu, nu*N) )

    Γ = -K̄*Γ
    Γ = vcat( zerorow, Γ )

    eye = blockkron( 1.0*I(N), Matrix( 1.0*I(nu) ) )
    eye = vcat( eye, zerorow )

    Γ = Γ + eye
    Γ = Γ[1:N*nu, 1:N*nu]

    return BlockArray{ mt }( Γ, [nu for i = 1:N], [nu for i = 1:N] )
end


function inputinitialpropagation( sys::StateSpace, K::AbstractArray, N::Integer )
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    blockGen = (-K*sys.A^i for i=0:(N-1))

    # The mortar function requires the passed array to have ndims=2, so reshape to
    # have an array with 2 dimensions even though there is only one column
    blocks = reshape( collect( blockGen ),  (N, 1) )

    return mortar( blocks )
end


"""
    hessian( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )

Form the complete Hessian for the constrained time-invariant LQR problem `clqr` using the fully
condensed form.
"""
function hessian( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    sys = getsystem( clqr )
    N   = clqr.N
    nx  = nstates( sys )
    nu  = ninputs( sys )

    Γ = prediction( sys, N )

    Q̅ = blockkron( 1.0*I(N), clqr.Qₖ )
    S̅ = blockkron( 1.0*I(N), clqr.S  )
    R̅ = blockkron( 1.0*I(N), clqr.R  )

    Q̅[Block( N, N )] = clqr.P
    S̅[Block( N, N )] = zeros( eltype( clqr.S ), size( clqr.S ) )

    # Pass the result through the Hermitian type to cleanup numerical errors that can occur
    # making it non-symmetric (they are on the order of 10e-17)
    return Hermitian( Γ'*Q̅*Γ + Γ'*S̅ + S̅'*Γ + R̅ )
end


"""
    linearcoefficients( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )

Form the vector of coefficients for the linear term of the fully condensed quadratic QP for the
CLQR problem defined by `clqr`.
"""
function linearcoefficients( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    sys = getsystem( clqr )
    N   = clqr.N
    nx  = nstates( sys )
    nu  = ninputs( sys )

    Γ = prediction( sys, N )
    Φ = initialpropagation( sys, N )

    Q̅ = blockkron( 1.0*I(N), clqr.Qₖ )
    S̅ = blockkron( 1.0*I(N), clqr.S  )

    Q̅[Block( N, N )] = clqr.P
    S̅[Block( N, N )] = zeros( eltype( clqr.S ), size( clqr.S ) )

    return Γ'*Q̅*Φ + S̅'*Φ
end


function inequalityconstraints( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    # For the initial computations we want the non-pre-stabilized system
    sys = getsystem( clqr, prestabilized = false )

    # Find the number of constraints and the system size
    N   = clqr.N
    nC  = size( clqr.E, 1 );    # Number of constraints
    n, m  = size( sys.B );

    # Get the condensed initial propagation matrix
    Φ = initialpropagation( sys, N )

    # Get the prediction matrix for the system
    Γ = prediction( sys, N )

    # Create the component matrices
    Ē = blockkron( 1.0*I(N), clqr.E )
    F̄ = blockkron( 1.0*I(N), clqr.F )

    # Compute the coefficient matrix of the linear inequality constraints
    G = Ē*Γ + F̄;

    # Compute the initial state matrix for the RHS of the constraints
    L = -Ē*Φ;

    if isprestabilized( clqr )
        # Get the pre-stabilized matrices
        sysₖ = getsystem( clqr )
        Γₖ   = inputprediction( sysₖ, clqr.K, N )
        Φₖ   = inputinitialpropagation( sysₖ, clqr.K, N )

        # Modify the initial state coefficients to be in the new input space
        L = L - G*Φₖ

        # Modify the coefficient matrix to be in the new input space
        G = G * Γₖ
    end

    # Compute the constant vector for the RHS
    g = blockkron( ones( Float64, (N) ), clqr.g )

    return G, L, g
end


function equalityconstraints( ::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    # The fully condensed form does not have any equality constraints from the problem
    return nothing
end
