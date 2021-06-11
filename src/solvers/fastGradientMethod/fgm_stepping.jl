"""
```julia
    ConstantStep()
```

Use a constant step size of ``( √L̅ - √̅μ̅  ) / ( √L̅ + √̅μ̅  )``, where ``L`` is the maximum eigenvalue
of the Hessian and ``μ`` is the minimum eigenvalue of the Hessian.

See also: [`VariableStep`](@ref)
"""
mutable struct ConstantStep <: AbstractStep
    β

    function ConstantStep()
        return new( 1.0 )
    end
end

function init!( step::ConstantStep, L, μ )
    step.β = ( sqrt( L ) - sqrt( μ ) ) / ( sqrt( L ) + sqrt( μ ) )
end

struct ConstantStepIterator{T}
    β::T
end

iterate( cs::ConstantStep ) = (cs.β, ConstantStepIterator{typeof( cs.β )}( cs.β ))
iterate( cs::ConstantStep, step::ConstantStepIterator ) = (cs.β, step)


"""
```julia
    VariableStep()
```

Use a variable step size  computed by finding the roots of a simple polynomial.

See Algorithm II.1 in [1] for more information.

[1] S. Richter, C. N. Jones, and M. Morari, ‘Computational Complexity Certification for Real-Time MPC With Input
    Constraints Based on the Fast Gradient Method’, IEEE Transactions on Automatic Control, vol. 57, no. 6,
    pp. 1391–1403, 2012, doi: 10.1109/TAC.2011.2176389.

See also: [`ConstantStep`](@ref)
"""
mutable struct VariableStep <: AbstractStep
    L
    μ
    func

    function VariableStep()
        return new( 0.0, 0.0, nothing )
    end
end

function init!( step::VariableStep, L, μ )
    step.L = L
    step.μ = μ

    # This let block creates a local copy of the variables for the closure, which
    # speeds up the computations (https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured)
    step.func = let μ = μ, L = L
        (α, αₙ) -> ( 1 - αₙ ) * α^2 + μ/L * αₙ - αₙ^2
    end
end

mutable struct VariableStepIterator{T} <: AbstractStep
    α::T
    αₙ::T
end

function iterate( vs::VariableStep )
    α = sqrt( vs.μ / vs.L )

    step = VariableStepIterator{typeof( α )}( α, α )

    return iterate( vs, step )
end

function iterate( vs::VariableStep, step::VariableStepIterator )
    step.α = step.αₙ

    # Search for the zero crossing inside (0,1)
    step.αₙ = find_zero( αₙ -> vs.func( step.α, αₙ ), (0, 1) )

    β = ( step.α * ( 1 - step.α ) ) / ( step.α^2 + step.αₙ )

    return (β, step)
end
