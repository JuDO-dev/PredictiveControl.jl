
# Import this because it is only used in conditionBound, and it has some methods that conflict with ControlSystems
import DescriptorSystems

function conditionBound( clqr::ConstrainedTimeInvariantLQR; L::Matrix = zeros(Float64, 1, 1) )
    sys = getsystem( clqr )
    nx  = nstates( sys )
    nu  = ninputs( sys )

    # The bound can only be computed if the predicted system is stable and there is no S matrix
    # So if those conditions aren't met, just use the exact condition number
    if( !isstable( sys ) )
        throw( DomainError( sys, "Condition number bound is only available for stable systems." ) )

    elseif( !iszero( clqr.S ) )
        throw( DomainError( clqr.S, "Condition number bound is only available for problems with a zero S matrix." ) )

    end

    if iszero( L )
        L = 1.0*I(nu)
    end

    # The adjoint system is actually a descriptor system, so we need to form one for our computations
    desc = DescriptorSystems.dss( sys.A, sys.B, sys.C, sys.D; Ts=sys.Ts )

    # This is the matrix symbol for the Hessian
    fullsys = L*( desc'*clqr.Qₖ*desc + desc'*clqr.K'*clqr.R + clqr.R*clqr.K*desc + clqr.R )*L'

    # Actually compute the condition number by using the H\_infty norm
    (maxeig, )    = DescriptorSystems.ghinfnorm( fullsys )
    (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( fullsys ) )

    # mineiginv is computed using the inverse system, so it actually is the inverse of the minimum eigenvalue of the original system
    return maxeig * mineiginv
end


function LinearAlgebra.cond( clqr::ConstrainedTimeInvariantLQR, reqType::Symbol = :Exact )

    if( reqType == :Exact || reqType == :exact)
        # Exact mode just computes the actual condition number for the Hessian of the
        # given problem
        H = hessian( clqr, PrimalFullCondensing )
        return LinearAlgebra.cond( Matrix( H ) )

    elseif( reqType == :Bound || reqType == :bound)
        return conditionBound( clqr )

    else
        error( "Unknown condition type. reqType must be either :Exact or :Bound.")
    end

end
