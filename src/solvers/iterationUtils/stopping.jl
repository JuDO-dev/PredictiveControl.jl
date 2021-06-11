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

function iterate( iter::StoppingConditionIterable )
    next = iterate( iter.iter )
    return dispatch( iter, next )
end

function iterate( iter::StoppingConditionIterable, (inst, next) )
    if inst == :halt
        return nothing
    end

    next = iterate( iter.iter, next )
    return dispatch( iter, next )
end


function dispatch( iter::StoppingConditionIterable, next )
    if next === nothing
        return nothing
    end

    nextInst = :continue

    for s in iter.condVec
        if s( next[2] )
            nextInst = :halt
        end
    end

    return (next[1], (nextInst, next[2]))
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
