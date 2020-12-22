
function sdpblockpreconditoner( H::HermitianBlockMatrix )
    model = JuMP.Model( COSMO.Optimizer )
    JuMP.set_silent( model )

    bs = blocksize( H )

    if( ( bs[1] != bs[2] ) || ( size( H, 1 ) != size( H, 2 ) ) )
        throw( DimensionMismatch( "Hessian H must be square" ) )
    end

    N  = bs[1]
    nu = size( H[Block(1, 1)], 1 )

    # The cholesky of the Hessian is needed for the constraints
    C = cholesky( Matrix(H) )

    E = zeros( GenericAffExpr{Float64,VariableRef}, N*nu, N*nu )

    for i in 1:N
        # Create the diagonal indices
        ind = ( ( nu*(i-1)+1 ):(nu*i) )

        # Create the blocks that go on the diagonal of the constraint matrix
        E[ind, ind] = @variable( model, [j=1:nu, k=1:nu], base_name="e$i" )
    end

    # Initialize the objective
    @variable( model, t )
    @objective( model, Min, t )

    # E should be PSD
    @constraint( model, E in PSDCone() )

    # The coefficient of E is the smallest eigenvalue of the preconditoned matrix
    @SDconstraint( model, Matrix(H) - E >= 0 )
    @SDconstraint( model, [E           Matrix(C.L);
                           Matrix(C.U) t*I(N*nu)] >= 0 )

    JuMP.optimize!( model )

    if( termination_status( model ) == MOI.OPTIMAL )
        sol = value.( E )
        obj = objective_value( model )

    elseif( termination_status( model ) == MOI.TIME_LIMIT && has_values( model ) )
        sol = value.( E )
        obj = objective_value( model )

    else
        # If we can't compute it using a pure diagonal structure, fall back on a full E matrix
        @warn "Unable to compute a preconditioning matrix using pure diagonal structure. Trying a full matrix."

        fullModel = JuMP.Model( COSMO.Optimizer )
        JuMP.set_silent( fullModel )

        # Initialize the objective
        @variable( fullModel, t )
        @objective( fullModel, Min, t )

        # E should be PSD
        @variable( fullModel, E[1:N*nu, 1:N*nu] )
        @constraint( fullModel, E in PSDCone() )

        # The coefficient of E is the smallest eigenvalue of the preconditoned matrix
        @SDconstraint( fullModel, Matrix(H) - E >= 0 )
        @SDconstraint( fullModel, [E           Matrix(C.L);
                                   Matrix(C.U) t*I(N*nu)] >= 0 )

        JuMP.optimize!( fullModel )

        if( termination_status( fullModel ) == MOI.OPTIMAL )
            sol = value.( E )
            obj = objective_value( fullModel )

        elseif( termination_status( fullModel ) == MOI.TIME_LIMIT && has_values( fullModel ) )
            sol = value.( E )
            obj = objective_value( fullModel )
        else
            error( "Unable to compute preconditioner using full matrix" )
        end
    end

    blockSol = BlockMatrix( sol, [nu for i=1:N], [nu for i=1:N] )
    precond  = BlockMatrix( zeros( N*nu, N*nu ), [nu for i=1:N], [nu for i=1:N] )

    for i = 1:N
        eig = eigen( blockSol[Block(i,i )] )

        # We need the square root of the inverse eigenvalues
        v = sqrt.( 1 ./ eig.values )

        # Reassemble the block
        precond[Block(i, i)] = eig.vectors * diagm( v ) * eig.vectors'
    end

    return precond
end


"""
    circulantblockpreconditioner( clqr::ConstrainedTimeInvariantLQR; variant::Symbol = :Strang, form::Symbol = :Symmetric, mat::Symbol = :Full )

Compute the circulant block preconditioner of a constrained LTI LQR problem. This preconditioner is similar
to the one computed using the SDP approach in [`sdpblockpreconditoner`](@ref), but is simpler to compute and
only works for LTI problems.

This will return the diagonal block used to form the preconditioner, not the entire preconditioning matrix.
By default, the block for the symmetric preconditioner ``L^{-1} H L^{-T}`` is returned, but the left/right
preconditioning block can be requested by setting `precondType` to either `:Left` or `:Right`.

"""
function circulantblockpreconditioner( clqr::ConstrainedTimeInvariantLQR; variant::Symbol = :Strang, form::Symbol = :Symmetric, mat::Symbol = :Full )
    sys = getsystem( clqr )

    # We need the solution to the DARE for the prestabilized system (using the original Q matrix)
    P, = ared( sys.A, sys.B, clqr.R, clqr.Q)

    # Form the Hessian block
    H = sys.B' * P * sys.B + clqr.R

    if variant == :Strang
        M = H
    elseif variant == :Chan
        @show E = dftmatrix( size( H, 1 ), unitary = true )

        @show D = Diagonal( E * H * E' )
        M = real( E' * D * E )
    else
        error( "Unknown variant specified. Valid types are :Strang and :Chan" )
    end

    # If a symmetric preconditioner is requested, use the lower-diagonal term of the Cholesky decomposition
    L = ( form == :Symmetric ) ? cholesky( Hermitian( M ) ).L : M

    if( mat == :Full)
        bs = ( ninputs(sys) for i=1:clqr.N )
        return BlockMatrix( kron( I(clqr.N), L ), collect( bs ), collect( bs ) )
    else
        return L
    end
end
