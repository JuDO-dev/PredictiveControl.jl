"""
```julia
    Conjugate( ϵ::T, scaled::Bool = true ) where {T <: Number}
```

Terminate the Fast Gradient Method when the conjugate residual value is below `ϵ`.
The value of `ϵ` is divided by the number of variables in the problem when `scaled`
is true, producing a scaled termination criteria.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Gradient`](@ref) and [`Best`](@ref)
"""
mutable struct Conjugate{T} <: AbstractStopCondition where {T <: Number}
    ϵ::T

    scale::Bool   # Should a scaled termination criteria be used
    n::Int        # Number of variables in the problem
    ϵₛ::T         # Scaled termination criteria

    function Conjugate( ϵ::T; scaled::Bool = true ) where {T <: Number}
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

function (cond::Conjugate)( state::FGMState )
    return compute_conjugate( state ) < cond.ϵₛ
end

function compute_conjugate( state::FGMState )
    return abs( state.xₙ'*state.∇xₙ + norm( state.∇xₙ, 1 ) )
end


"""
```julia
    Gradient( ϵ::T, scaled::Bool = true ) where {T <: Number}
```

Terminate the Fast Gradient Method when the gradient value is below `ϵ`.
The value of `ϵ` is divided by the number of variables in the problem when `scaled`
is true, producing a scaled termination criteria.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Conjugate`](@ref) and [`Best`](@ref)
"""
mutable struct Gradient{T} <: AbstractStopCondition where {T <: Number}
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

function (cond::Gradient)( state::FGMState )
    return abs( cond.c * norm( cond.L* ( state.y - state.xₙ ) )^2 ) < cond.ϵₛ
end


"""
```julia
    Best( ϵ₁::T, ϵ₂:T = ϵ₁; scaled::Bool = true ) where {T <: Number}
```

Terminate the Fast Gradient Method when either the gradient value is below `ϵ₁` or the
conjugate residual value is below `ϵ₂`. `ϵ₁` and `ϵ₂` are divided by the number of variables
in the problem to produce a scaled termination criteria when `scaled` is true.

This criteria is based on the results in §6.4 of [1].

[1] S. Richter, ‘Computational complexity certification of gradient methods for real-time model predictive control’, Doctoral Thesis, ETH Zurich, 2012.

See also: [`Gradient`](@ref) and [`Conjugate`](@ref)
"""
struct Best <: AbstractStopCondition
    grad::Gradient
    conj::Conjugate

    function Best( ϵ₁::T, ϵ₂::T = ϵ₁; scaled::Bool = true ) where {T <: Number}
        return new( Gradient( ϵ₁; scaled = scaled ), Conjugate( ϵ₂; scaled = scaled ) )
    end
end

function init!( cond::Best, n, L, μ )
    init!( cond.grad, n, L, μ )
    init!( cond.conj, n, L, μ )
end

function (cond::Best)( state::FGMState )
    return cond.grad( state ) || cond.conj( state )
end
