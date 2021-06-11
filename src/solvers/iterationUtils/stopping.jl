"""
    StopCondition

An abstract type that all stopping conditions for the iterative method
should inherit from.

Each type should define the () operator in itself, and return true from that
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

function iterate( iter::StoppingConditionIterable, (instruction, state) )
    if instruction == :halt
        return nothing
    end

    next = iterate( iter.iter, state )
    return dispatch( iter, next )
end

function dispatch( iter::StoppingConditionIterable, next )
    if next === nothing
        return nothing
    end

    stop = false

    for s in iter.condVec
        stop |= s( next[1] )

        stop && break
    end

    return next[1], (stop ? :halt : :continue, next[2])
end


"""
    halt( iter::I, condVec::Vector{SC} ) where {I, SC}

Construct an iterator that will check the state of `iter` against the stopping conditions
`condVec`.
"""
halt( iter::I, condVec::Vector{SC} ) where {I, SC} = StoppingConditionIterable{I, eltype(condVec)}( iter, condVec )

"""
    halt( iter::I, cond::SC ) where {I, SC <: AbstractStopCondition}

Construct an iterator that will check the state of `iter` against the stopping condition
`condVec`.
"""
halt( iter::I, cond::SC ) where {I, SC <: AbstractStopCondition} = StoppingConditionIterable{I, SC}( iter, [cond] )
