using Base.Iterators
using PredictiveControl
using Test

import Base: iterate

struct TestIter end

mutable struct TestIterState
    iter::Int
    flag1::Vector{Int}
    flag2::Vector{Int}
    flag3::Vector{Int}

    function TestIterState()
        return new( 0, Int[], Int[], Int[] )
    end
end

function iterate( iter::TestIter )
    state = TestIterState()
    return (state, state)
end

function iterate( iter::TestIter, state::TestIterState )
    state.iter = state.iter + 1
    return (state, state)
end

function applyflag1( state::TestIterState )
    push!( state.flag1, state.iter )
end

function applyflag2( state::TestIterState )
    push!( state.flag2, state.iter )
end

function applyflag3( state::TestIterState )
    push!( state.flag3, state.iter )
end

let iter1 = TestIter()
    iter1 = take( iter1, 10 )
    iter1 = enumerate( iter1 )
    iter1 = PredictiveControl.FGM.apply( iter1, applyflag1 )

    x = nothing
    for y in iter1
        x = y
    end

    (_, finalstate) = x

    # One element extra because it started with 1 element before the 10 iterations
    @test length( finalstate.flag1 ) == 10
    @test length( finalstate.flag2 ) == 0
    @test length( finalstate.flag3 ) == 0
end

let iter2 = TestIter()
    iter2 = take( iter2, 10 )
    iter2 = enumerate( iter2 )
    iter2 = PredictiveControl.FGM.apply( iter2, applyflag1 )
    iter2 = PredictiveControl.FGM.apply( iter2, applyflag2, 2 )

    x = nothing
    for y in iter2
        x = y
    end

    (_, finalstate) = x

    @test length( finalstate.flag1 ) == 10
    @test length( finalstate.flag2 ) == 5
    @test length( finalstate.flag3 ) == 0
end

let iter3 = TestIter()
    iter3 = take( iter3, 10 )
    iter3 = enumerate( iter3 )
    iter3 = PredictiveControl.FGM.apply( iter3, applyflag1 )
    iter3 = PredictiveControl.FGM.apply( iter3, applyflag2, 2 )
    iter3 = PredictiveControl.FGM.apply( iter3, applyflag3, 5 )

    x = nothing
    for y in iter3
        x = y
    end

    (_, finalstate) = x

    @test length( finalstate.flag1 ) == 10
    @test length( finalstate.flag2 ) == 5
    @test length( finalstate.flag3 ) == 2
end
