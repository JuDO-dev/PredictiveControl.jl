"""
    StopCondition

An abstract type that all stopping conditions for the Fast Gradient Method
should inherit from.
"""
abstract type StopCondition end

"""
    Iteration( n::Int )

Terminate the solver after `n` iterations.
"""
mutable struct Iteration <: StopCondition
    n::Int
end

function init!( ::Iteration, n, L, μ )
end

function (cond::Iteration)( i, r, x, y, xₙ, yₙ )
    return cond.n <= i
end


"""
    Conjugate( ϵ::T, scaled::Bool = true ) where {T <: Number}

Terminate the Fast Gradient Method when the conjugate residual value is below `ϵ`.
The value of `ϵ` is divided by the number of variables in the problem when `scaled`
is true, producing a scaled termination criteria.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Gradient`](@ref) and [`Best`](@ref)
"""
mutable struct Conjugate{T} <: StopCondition where {T <: Number}
    ϵ::T

    scale::Bool   # Should a scaled termination criteria be used
    n::Int        # Number of variables in the problem
    ϵₛ::T         # Scaled termination criteria

    function Conjugate( ϵ::T, scaled::Bool = true ) where {T <: Number}
        return new{T}( ϵ, scaled, 1, ϵ )
    end
end

function init!( cond::Conjugate, n, L, μ )
    cond.n = n

    if( cond.scale )
        cond.ϵₛ = cond.ϵ * 1/n
    else
        cond.ϵₛ = cond.ϵ
    end
end

function (cond::Conjugate)( i, r, x, y, xₙ, yₙ )
    return abs( xₙ'*r + norm( r, 1 ) ) < cond.ϵₛ
end


"""
    Gradient( ϵ::T, scaled::Bool = true ) where {T <: Number}

Terminate the Fast Gradient Method when the gradient value is below `ϵ`.
The value of `ϵ` is divided by the number of variables in the problem when `scaled`
is true, producing a scaled termination criteria.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Conjugate`](@ref) and [`Best`](@ref)
"""
mutable struct Gradient{T} <: StopCondition where {T <: Number}
    ϵ::T
    L::T
    μ::T
    c::T

    scale::Bool   # Should a scaled termination criteria be used
    n::Int        # Number of variables in the problem
    ϵₛ::T         # Scaled termination criteria


    function Gradient( ϵ::T; scaled::Bool = true ) where {T <: Number}
        return new{T}( ϵ, scaled, 0.0, 0.0, scaled, 1, ϵ )
    end
end

function init!( cond::Gradient, n, L, μ )
    cond.μ = μ     #
    cond.L = L     #
    cond.c = 0.5 * ( 1/cond.μ - 1/cond.L )  # Coefficient of the computation

    cond.n = n

    if( cond.scale )
        cond.ϵₛ = cond.ϵ * 1/n
    else
        cond.ϵₛ = cond.ϵ
    end
end

function (cond::Gradient)( i, r, x, y, xₙ, yₙ )
    return abs( cond.c * norm( cond.L* ( y - xₙ ) )^2 ) < cond.ϵₛ
end


"""
    Best( ϵ₁::T, ϵ₂:T = ϵ₁; scaled::Bool = true ) where {T <: Number}

Terminate the Fast Gradient Method when either the gradient value is below `ϵ₁` or the
conjugate residual value is beloe `ϵ₂`. `ϵ₁` and `ϵ₂` are divided by the number of variables
in the problem to produce a scaled termination criteria when `scaled` is true.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Gradient`](@ref) and [`Conjugate`](@ref)
"""
struct Best <: StopCondition
    grad::Gradient
    conj::Conjugate

    function Best( ϵ₁::T, ϵ₂::T = ϵ₁; scaled::Bool = true ) where {T <: Number}
        return new( Gradient( ϵ₁, scaled ), Conjugate( ϵ₂, scaled ) )
    end
end

function init!( cond::Best, n, L, μ )
    init!( cond.grad, n, L, μ )
    init!( cond.conj, n, L, μ )
end

function (cond::Best)( i, r, x, y, xₙ, yₙ )
    return cond.grad( i, r, x, y, xₙ, yₙ ) || cond.conj( i, r, x, y, xₙ, yₙ )
end


"""
    AbstractStep

An abstract super type that is used to specify the step length computation in the
Fast Gradient Method. Implementations of the step length computation should define
a type that inherits from this abstract type.
"""
abstract type AbstractStep end

"""
    ConstantStep{T}() where {T <: Number}

Use a constant step size of ``( √L̅ - √̅μ̅  ) / ( √L̅ + √̅μ̅  )``, where ``L`` is the maximum eigenvalue
of the Hessian and ``μ`` is the minimum eigenvalue of the Hessian.

See also: [`VariableStep`](@ref)
"""
mutable struct ConstantStep{T} <: AbstractStep
    beta::T

    function ConstantStep{T}() where {T <: Number}
        new( zero(T) )
    end
end

function init!( step::ConstantStep{T}, L, μ ) where {T <: Number}
    step.beta = ( sqrt(L) - sqrt(μ) ) / ( sqrt(L) + sqrt(μ) )
end

function (step::ConstantStep)()
    return step.beta
end


"""
    VariableStep{T}() where {T <: Number}

Use a variable step size  computed by finding the roots of a simple polynomial.

See Algorithm II.1 in [1] for more information.

[1] S. Richter, C. N. Jones, and M. Morari, ‘Computational Complexity Certification for Real-Time MPC With Input Constraints Based on the Fast Gradient Method’, IEEE Transactions on Automatic Control, vol. 57, no. 6, pp. 1391–1403, 2012, doi: 10.1109/TAC.2011.2176389.

See also: [`ConstantStep`](@ref)
"""
mutable struct VariableStep{T} <: AbstractStep
    α::T
    αₙ::T
    func::Function

    function VariableStep{T}() where {T <: Number}
        return new( zero(T), zero(T), x -> zero(T) )
    end
end

function init!( step::VariableStep{T}, L, μ ) where {T <: Number}
    step.α  = sqrt( μ / L )
    step.αₙ = step.α

    # This let block creates a local copy of the variables for the closure, which
    # speeds up the computations (https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured)
    step.func = let μ = μ, L = L, α = step.α
        αₙ -> ( 1 - αₙ ) * α^2 + μ/L * αₙ - αₙ^2
    end
end

function (step::VariableStep)()
    step.α = step.αₙ

    # Search for the zero crossing inside (0,1)
    step.αₙ = find_zero( step.func, (0, 1) )

    return ( step.α * ( 1 - step.α ) ) / ( step.α^2 + step.αₙ )
end


"""
    fastgradientmethod( H::Matrix{T}, b::Vector{T}, proj::Function; x₀::Vector{T} = zeros( T, size( b ) ),
                                                                    L::Union{Nothing, T} = nothing,
                                                                    μ::Union{Nothing, T} = nothing,
                                                                    step::AbstractStep = ConstantStep{T}(),
                                                                    stopconditions::Vector{SC} = [Best(1e-4)]
                           ) where {T <: Number, SC <: StopCondition}

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
function fastgradientmethod( H::Matrix{T}, b::Vector{T}, proj::Function; x₀::Vector{T} = zeros( T, size( b ) ),
                                                                         L::Union{Nothing, T} = nothing,
                                                                         μ::Union{Nothing, T} = nothing,
                                                                         step::AbstractStep = ConstantStep{T}(),
                                                                         stopconditions::Vector{SC} = [Best(1e-4)]
                           ) where {T <: Number, SC <: StopCondition}

    # Compute the lipschitz and convexity parameter if they aren't provided
    if isnothing( L ) || isnothing( μ )
        eig = eigvals( H )

        isnothing( L ) && ( L = maximum( eig ) )
        isnothing( μ ) && ( μ = minimum( eig ) )
    end

    x  = x₀
    xₙ = x₀
    y  = x₀
    yₙ = x₀

    n = size( x, 1 )

    for cond in stopconditions
        init!( cond, n, L, μ )
    end

    init!( step, L, μ )

    iter = 0

    while true
        iter = iter + 1

        # Copy the previous values into the current values
        x = xₙ
        y = yₙ

        # Compute the gradient
        ∇y = H*y + b

        # Compute the new step
        t = y - (1/L) * ∇y

        # Project into the feasible space
        xₙ = proj( t )

        # Compute the step length and apply it
        β  = step()
        yₙ = xₙ + β*(xₙ - x)

        # Compute the residual of the next point for the stopping condition computation
        r = H*xₙ + b

        # Test the stop conditions
        stop = false
        for cond in stopconditions
            cond( iter, r, x, y, xₙ, yₙ ) && ( stop = true )
        end

        if stop
            break
        end
    end

    return xₙ, iter
end
