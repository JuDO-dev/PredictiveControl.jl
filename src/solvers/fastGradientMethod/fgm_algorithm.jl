mutable struct FGMState{VT, IT}
    x::VT
    xₙ::VT
    y::VT
    yₙ::VT
    ∇y::VT
    ∇xₙ::VT
    t::VT
    stepstate::IT
end

struct FastGradientMethodIterator{MT, VT, IT, T}
    H::MT
    b::VT
    x₀::VT
    proj::Function
    step::IT
    L::T
end

function iterate( fgm::FastGradientMethodIterator )
    VT = typeof( fgm.x₀ )
    ET = eltype( fgm.x₀ )

    len = length( fgm.x₀ )
    zer = fill( zero(ET), (len, ) )

    # Setup the step length computation iterator
    (β, stepstate) = iterate( fgm.step )

    IT = typeof( stepstate )

    state = FGMState{VT, IT}( fgm.x₀, fgm.x₀, fgm.x₀, fgm.x₀, zer, zer, zer, stepstate )

    # Run the first pass
    return iterate( fgm, state )
end

function iterate( fgm::FastGradientMethodIterator, state::FGMState )
    # Copy the previous values into the current values
    state.x = state.xₙ
    state.y = state.yₙ

    # Compute the gradient
    state.∇y = fgm.H*state.y + fgm.b

    # Compute the new step
    state.t = state.y - (1/fgm.L) * state.∇y

    # Project into the feasible space
    state.xₙ = fgm.proj( state.t )

    # Compute the step length and apply it
    (β, state.stepstate) = iterate( fgm.step, state.stepstate )

    state.yₙ = state.xₙ + β*( state.xₙ - state.x )

    # Compute the residual of the next point for the stopping condition computation
    state.∇xₙ = fgm.H*state.xₙ + fgm.b

    return (state, state)
end


"""
function fastgradientmethod( H::AbstractMatrix{T}, b::AbstractVector{T}, proj::Function; x₀::AbstractVector{T} = zeros( T, size( b ) ),
                                                                                         L::Union{Nothing, T} = nothing,
                                                                                         μ::Union{Nothing, T} = nothing,
                                                                                         step::AbstractStep = ConstantStep(),
                                                                                         stopconditions::Vector{SC} = [Best(1e-4)],
                                                                                         maxiter::Integer = 100
                           ) where {T <: Number, SC <: AbstractStopCondition}

Compute the solution to the quadratic program ``min xᵀHx + bᵀx s.t. x ∈ χ`` where the set constraint
``χ`` is supplied as a projection operation that maps a value of ``x`` to a value inside the set ``χ``.
The projection operator is specified as the function `proj`, which must take a single argument which is
a vector of type `T` (the current iterate) and return a vector of type `T` (the projection of the iterate
onto the set ``Χ``).

The behavior of the algorithm can be modified using the keyword arguments to set various algorithm parameters:
* x₀ - The initial iterate
* L - User-specified value for the maximum eigenvalue of `H`. If not specified, then the initial phase of the
      algorithm will perform an eigenvalue decomposition on `H` and set `L` to its maximum eigenvalue.
* μ - User-specified value for the minimum eigenvalue of `H`. If not specified, then the initial phase of the
      algorithm will perform an eigenvalue decomposition on `H` and set `L` to its minimum eigenvalue.
* step - How the step length should be computed at each iteration.
* stopconditions - A vector containing the stopping conditions that the algorithm uses to determine convergence.
                   A boolean OR of all the stopping conditions will be used, so when the first one is met, the
                   algorithm will terminate.
"""
function fastgradientmethod( H::AbstractMatrix{T}, b::AbstractVector{T}, proj::Function; x₀::AbstractVector{T} = zeros( T, size( b ) ),
                                                                                         L::Union{Nothing, T} = nothing,
                                                                                         μ::Union{Nothing, T} = nothing,
                                                                                         step::AbstractStep = ConstantStep(),
                                                                                         stopconditions::Union{Vector{SC}, Nothing} = [Best(1e-4)],
                                                                                         maxiter::Integer = 100
                           ) where {T <: Number, SC <: AbstractStopCondition}

    # Compute the lipschitz and convexity parameter if they aren't provided
    if isnothing( L ) || isnothing( μ )
        eig = eigvals( H )

        isnothing( L ) && ( L = maximum( eig ) )
        isnothing( μ ) && ( μ = minimum( eig ) )
    end

    n = size( b, 1 )

    init!( step, L, μ )

    iter = 0

    # Create the actual iterator that we use
    MT = typeof( H )
    VT = typeof( b )
    IT = typeof( step )
    LT = typeof( L )

    iter = FastGradientMethodIterator{MT, VT, IT, LT}( H, b, x₀, proj, step, L )

    # Setup the stopping conditions
    iter = take( iter, maxiter )

    if stopconditions != nothing
        for cond in stopconditions
            init!( cond, n, L, μ )
        end

        iter = halt( iter, stopconditions )
    end

    iter = enumerate(iter)

    ((_, finalstate), numiter) = loop( iter )

    return (finalstate.xₙ, numiter)
end
