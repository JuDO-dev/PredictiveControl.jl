"""
    FGMSolver
"""
mutable struct FGMSolver
    step::FGM.AbstractStep
    cond::Vector{FGM.AbstractStopCondition}

    function FGMSolver( step::FGM.AbstractStep = FGM.ConstantStep{T}(), sc::Vector{FGM.AbstractStopCondition} = [FGM.Best(1e-4)] )
        solver = new( step, sc )

        return solver
    end
end


function createsolver!( ::FGMSolver )

end


function updatesolver!( ::FGMSolver )

end


function solve!( ::FGMSolver  )

end
