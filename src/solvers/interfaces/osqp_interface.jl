"""
    OSQPSolver
"""
mutable struct OSQPSolver
    options     # Dictionary of options to pass to OSQP

    model::OSQP.Model  # The OSQP solver itself

    function OSQPSolver( opts )
        solver = new()

        solver.options = opts

        return solver
    end
end


function createsolver!( solver::OSQPSolver )

    solver.model = OSQP.Model()

    options = Dict( :verbose => false )

    OSQP.setup!( solver.model; P = sparse( triu( 1.0*I(m) ) ),
                               q = zeros( Float64, (m) ),
                               A = sparse( G ),
                               u = g,
                               solver.options... )

end


function updatesolver!( solver::OSQPSolver )

end


function solve!( solver::OSQPSolver )

    results = OSQP.solve!( solver.model )

end
