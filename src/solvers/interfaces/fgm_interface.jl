"""
    FGMSolver
"""
mutable struct FGMSolver
    step::FGM.AbstractStep
    cond::Vector{FGM.AbstractStopCondition}
    maxiter::Int

    function FGMSolver(; step::FGM.AbstractStep = FGM.ConstantStep{T}(), sc::Vector{FGM.AbstractStopCondition} = [FGM.Best(1e-4)], maxiter = 100 )
        solver = new( step, sc, maxiter )

        return solver
    end
end


function createsolver!( ::FGMSolver )

end


function updatesolver!( ::FGMSolver )

end


function solve!( ::FGMSolver )

end
