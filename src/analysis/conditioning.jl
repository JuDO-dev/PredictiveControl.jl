
# Import this because it is only used in conditionBound, and it has some methods that conflict with ControlSystems
import DescriptorSystems

function conditionBound( clqr::ConstrainedTimeInvariantLQR; L::Matrix = zeros(Float64, 1, 1), samplingPoints::Int = 0 )
    sys = getsystem( clqr, prestabilized = true )
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

    if samplingPoints == 0
        # The adjoint system is actually a descriptor system, so we need to form one for our computations
        #z = DescriptorSystems.dss( DescriptorSystems.rtf(:z; Ts=sys.Ts) )
        #desc = z * DescriptorSystems.dss( sys.A, sys.B, sys.C, sys.D; Ts=sys.Ts )

        desc = DescriptorSystems.dss( sys.A, sys.B, sys.C, sys.D; Ts=sys.Ts )

        #crossterm = desc'*clqr.K'*clqr.R
        crossterm = clqr.R*clqr.K*desc

        (maxeig, )    = DescriptorSystems.ghinfnorm( crossterm )
        (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( crossterm ) )

        fullcrossterm = crossterm + crossterm'

        (maxeig, )    = DescriptorSystems.ghinfnorm( fullcrossterm )
        (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( fullcrossterm ) )


        # This is the matrix symbol for the Hessian
        fullsys = L*( desc'*clqr.Qₖ*desc - crossterm - crossterm' + clqr.R )*L'

        # Actually compute the condition number by using the H\_infty norm
        (maxeig, )    = DescriptorSystems.ghinfnorm( fullsys )
        (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( fullsys ) )

        # mineiginv is computed using the inverse system, so it actually is the inverse of the minimum eigenvalue of the original system
        @show maxeig * mineiginv

        # This is the matrix symbol for the Hessian
        fullsys = L*( desc'*clqr.Qₖ*desc + crossterm + crossterm' + clqr.R )*L'

        # Actually compute the condition number by using the H\_infty norm
        (maxeig, )    = DescriptorSystems.ghinfnorm( fullsys )
        (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( fullsys ) )

        # mineiginv is computed using the inverse system, so it actually is the inverse of the minimum eigenvalue of the original system
        maxeig * mineiginv

        # This is the matrix symbol for the Hessian
        fullsys = L*( desc'*clqr.Qₖ*desc + clqr.R )*L'

        # Actually compute the condition number by using the H\_infty norm
        (maxeig, )    = DescriptorSystems.ghinfnorm( fullsys )
        (mineiginv, ) = DescriptorSystems.ghinfnorm( inv( fullsys ) )

        # mineiginv is computed using the inverse system, so it actually is the inverse of the minimum eigenvalue of the original system
        return maxeig * mineiginv
    else
        # We need the actual transfer function matrix as an evaluatable function
        pᵧ(z) = z*(sys.C*inv( z*I(nx) - sys.A )*sys.B + sys.D)
        f(z)  = L*( pᵧ(z)'*clqr.Qₖ*pᵧ(z) - clqr.R*clqr.K*pᵧ(z) - pᵧ(z)'*clqr.K'*clqr.R + clqr.R )*L'

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
