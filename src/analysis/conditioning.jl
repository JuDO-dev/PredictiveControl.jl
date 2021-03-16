
function conditionBound( clqr::ConstrainedTimeInvariantLQR, samplingPoints::Integer = 100; L::Matrix = zeros(Float64, 1, 1) )
    sys = getsystem( clqr )
    nx  = nstates( sys )
    nu  = ninputs( sys )

    # The bound can only be computed if the predicted system is stable and there is no S matrix
    # So if those conditions aren't met, just use the exact condition number
    if( !isstable( sys ) )
        throw( DomainError( sys, "Condition number bound is only available for stable systems." ) )

    elseif( !iszero( clqr.S ) )
        throw( DomainError( clqr.S, "Condition number bound is only available for problems with a zero S matrix." ) )

    elseif( samplingPoints < 1 )
        throw( DomainError( samplingPoints, "At least one sampling point must be used" ) )
    end

    if iszero( L )
        L = 1.0*I(nu)
    end

    # We need the actual transfer function matrix as an evaluatable function
    tf(z) = sys.C*inv( z*I(nx) - sys.A )*sys.B + sys.D

    # This is the matrix symbol for the Hessian
    f(z) = L*( tf(z)'*clqr.Qₖ*tf(z) + clqr.R )*L'

    # Generator for the points around the unit circle to sample at
    T = ( exp( im*( -π/2 + 2*π*i/samplingPoints ) ) for i=1:samplingPoints )

    # Generator for the matrices to explore for their eigenvalues
    tfm = ( f(z) for z in T )

    # Compute the eigenvalues of the sampled transfer function matrices
    mate = eigvals.( tfm )
    eigs = collect( Iterators.flatten( mate ) )
    eigs = abs.( eigs )

    sort!( eigs )

    return eigs[end] / eigs[1]
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
