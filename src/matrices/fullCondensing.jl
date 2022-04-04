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
    initialpropagation( sys::StateSpace, N::Integer; ive::Bool = false )

Form the matrix ``Φ`` that propagates the initial condition ``x₀`` in the linear system `sys` across the horizon `N`.
When `ive` is true, the initial condition ``x₀`` is embedded in the resulting state vector

# Extended help

When `ive` is false, this matrix consists of a column of the blocks ``A^i`` where ``i`` is the row number, forming the matrix
```math
⎡ A  ⎤
  ⎢ A² ⎥
  ⎢ A³ ⎥
  ⎢ A⁴ ⎥
  ⎣ ⋮  ⎦
```

When `ive` is true, this matrix consists of a column of the blocks ``A^i-1`` where ``i`` is the row number, forming the matrix
```math
⎡ I  ⎤
  ⎢ A² ⎥
  ⎢ A² ⎥
  ⎢ A³ ⎥
  ⎣ ⋮  ⎦
```

It is used to compute the component of the state vector for a horizon that is caused by the intial state of the system using
the equation ``x = Φx₀``.

The full state vector for a horizon can be computed as ``x = Γu + Φx₀``, where ``Γ`` is the prediction matrix
from [`prediction`](@ref).
"""
function initialpropagation( sys::StateSpace, N::Integer; ive::Bool = false )
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    sub = ive ? 1 : 0

    blockGen = (sys.A^(i-sub) for i=1:N)

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
    sysₖ = getsystem( clqr, prestabilized = true )
    N    = clqr.N
    nₓ   = nstates( sysₖ )
    nᵤ   = ninputs( sysₖ )

    Iₙ = 1.0*I(N)
    Γₖ = prediction( sysₖ, N )

    Q = clqr.Q
    S = clqr.S
    K = clqr.K
    R = clqr.R

    Qₖ = Q - K'*S' - S*K + K'*R*K
    Sₖ = S - K'*R

    Q̄ₖ = blockkron( Iₙ, Qₖ )
    S̄ₖ = blockkron( Iₙ, Sₖ )
    R̄  = blockkron( Iₙ, R  )

    Q̄ₖ[Block( N, N )] = clqr.P
    S̄ₖ[Block( N, N )] = zeros( eltype( clqr.S ), size( clqr.S ) ) - K'*R

    # Pass the result through the Hermitian type to cleanup numerical errors that can occur
    # making it non-symmetric (they are on the order of 10e-17)
    return Hermitian( Γₖ'*Q̄ₖ*Γₖ + Γₖ'*S̄ₖ + S̄ₖ'*Γₖ + R̄  )
end

function hessian_shifted( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    sysₖ = getsystem( clqr, prestabilized = true )
    mt   = eltype( sysₖ.A )
    N    = clqr.N
    N₁   = N+1
    nₓ   = nstates( sysₖ )
    nᵤ   = ninputs( sysₖ )

    Iₙ  = 1.0*I(N)
    Iₙ₁ = 1.0*I(N₁)

    # Form the prediction matrix with the initial value embedded
    Γₖ = prediction( sysₖ, N₁ )
    Γ̃ₖ = BlockArray{ mt }( zeros( mt, nₓ*N₁, nᵤ*N ), [nₓ for i = 1:N₁], [nᵤ for i = 1:N] )
    [Γ̃ₖ[Block(i, j)] = Γₖ[Block(i-1, j)] for i=2:N₁, j=1:N]

    Q̄ₖ = blockkron( Iₙ₁,  clqr.Q )
    S̄  = blockkron(  Iₙ,  clqr.S )
    R̄  = blockkron(  Iₙ,  clqr.R )
    K̄  = blockkron(  Iₙ, -clqr.K )

    Q̄ₖ[Block( N₁, N₁ )] = clqr.P
    S̄ = vcat( S̄, zeros( eltype( clqr.S ), (nₓ, N*nᵤ ) ) )
    K̄ = hcat( K̄, zeros( eltype( clqr.K ), (N*nᵤ, nₓ ) ) )

    # Pass the result through the Hermitian type to cleanup numerical errors that can occur
    # making it non-symmetric (they are on the order of 10e-17)
    return Hermitian( Γ̃ₖ'*(Q̄ₖ + K̄'*R̄*K̄ + S̄ *K̄ + K̄'*S̄' )*Γ̃ₖ + Γ̃ₖ'*(K̄' *R̄ + S̄ ) + (R̄ *K̄ + S̄')*Γ̃ₖ + R̄  )
end


"""
    linearcoefficients( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )

Form the vector of coefficients for the linear term of the fully condensed quadratic QP for the
CLQR problem defined by `clqr`.
"""
function linearcoefficients( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    sysₖ = getsystem( clqr, prestabilized = true )
    N    = clqr.N
    nx   = nstates( sysₖ )
    nu   = ninputs( sysₖ )

    Iₙ = 1.0*I(N)
    Γₖ = prediction( sysₖ, N )
    Φₖ = initialpropagation( sysₖ, N )

    Q̄ₖ = blockkron( Iₙ,  clqr.Qₖ )
    S̄  = blockkron( Iₙ,  clqr.S  )
    R̄  = blockkron( Iₙ,  clqr.R  )
    K̄  = blockkron( Iₙ, -clqr.K  )

    Q̄ₖ[Block( N, N )] = clqr.P
    S̄[Block( N, N )]  = zeros( eltype( clqr.S ), size( clqr.S ) )

    # Temporary product used in two places
    T = 1.0*I(N*nu) + Γₖ'*K̄'

    return ((Γₖ'*S̄ + T*R̄')*K̄ + T*S̄' + Γₖ'*Q̄ₖ')*Φₖ
end


function inequalityconstraints( clqr::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    sysₖ = getsystem( clqr, prestabilized = true )

    mt   = eltype( sysₖ.A )
    N    = clqr.N
    nx   = nstates( sysₖ )
    nu   = ninputs( sysₖ )
    Iₙ   = 1.0*I(N)

    # Get the condensed initial propagation matrix
    Φ̃ₖ = initialpropagation( sysₖ, N, ive = true )

    # Form the prediction matrix with the initial value embedded
    Γₖ = prediction( sysₖ, N )
    Γ̃ₖ = BlockArray{ mt }( zeros( mt, nx*N, nu*N ), [nx for i = 1:N], [nu for i = 1:N] )
    [Γ̃ₖ[Block(i, j)] = Γₖ[Block(i-1, j)] for i=2:N, j=1:N]

    # Create the component matrices
    Ē  = blockkron( Iₙ, clqr.E )
    F̄  = blockkron( Iₙ, clqr.F )
    K̄  = blockkron( Iₙ, clqr.K  )

    # Common matrix
    # The minus sign here comes from the fact we have a negative feedback controller (u = -Kx)
    T = Ē - F̄ *K̄

    # Compute the coefficient matrix of the linear inequality constraints
    G = T*Γ̃ₖ + F̄

    # Compute the initial state matrix for the RHS of the constraints
    L = -T*Φ̃ₖ;

    # Compute the constant vector for the RHS
    g = blockkron( ones( Float64, (N) ), clqr.g )

    return G, L, g
end


function equalityconstraints( ::ConstrainedTimeInvariantLQR, ::Type{PrimalFullCondensing} )
    # The fully condensed form does not have any equality constraints from the problem
    return nothing
end
