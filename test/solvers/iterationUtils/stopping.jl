using Base.Iterators
using PredictiveControl
using Test

import Base: iterate

struct TestIter end

mutable struct TestIterState
    iter::UInt
end

function iterate( iter::TestIter )
    state = TestIterState( 0 )
    return (state, state)
end

function iterate( iter::TestIter, state::TestIterState )
    state.iter = state.iter + 1
    return (state, state)
end

struct TestStoppingCondition <: PredictiveControl.FGM.AbstractStopCondition end

function (t::TestStoppingCondition)( state::TestIterState )
    return state.iter > 4
end

iter = TestIter()
iter = PredictiveControl.FGM.halt( iter, TestStoppingCondition() )
iter = take( iter, 10 )
iter = enumerate( iter )

x = nothing

for y in iter
    global x = y
end

(_, finalstate) = x

# It should have ended after 5 iterations now
@test finalstate.iter == 5
