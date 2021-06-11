"""
```julia
    AbstractStopCondition
```

An abstract type that all stopping conditions for the iterative method
should inherit from.

Each type should define the () operator on itself, and return true from that
method when the stopping condition is met, and false if the condition is not
met.
"""
abstract type AbstractStopCondition end



struct StoppingConditionIterable{I, SC}
    iter::I
    condVec::Vector{SC}
end

function iterate( iter::StoppingConditionIterable, args... )
    next = iterate( iter.iter, args... )

    if next === nothing
        return nothing
    end

    for s in iter.condVec
        if s( next[2] )
            return nothing
        end
    end

    return next
end

"""
```julia
    halt( iter::I, condVec::Vector{SC} ) where {I, SC}
```

Construct an iterator that will check the state of `iter` against the stopping conditions
`condVec`.
"""
halt( iter::I, condVec::Vector{SC} ) where {I, SC} = StoppingConditionIterable{I, eltype(condVec)}( iter, condVec )

"""
```julia
    halt( iter::I, cond::SC ) where {I, SC <: AbstractStopCondition}
```

Construct an iterator that will check the state of `iter` against the stopping condition
`condVec`.
"""
halt( iter::I, cond::SC ) where {I, SC <: AbstractStopCondition} = StoppingConditionIterable{I, SC}( iter, [cond] )
