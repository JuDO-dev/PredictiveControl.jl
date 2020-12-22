
struct ConstrainedTimeInvariantLQR{T <: Number}
    sys::AbstractStateSpace # The predicted system
    N::Integer              # The horizon length
    Q::AbstractMatrix{T}    # The state weight matrix
    Qₖ::AbstractMatrix{T}   # The Q weighting matrix taking into account the prestabilizing controller
    R::AbstractMatrix{T}    # The input weight matrix
    P::AbstractMatrix{T}    # The terminal state weight matrix
    S::AbstractMatrix{T}    # The cross-term weight matrix
    K::AbstractMatrix{T}    # A prestabilizing controller (if any) that forms A-BK
    E::AbstractMatrix{T}    # The stage state constraint coefficient matrix
    F::AbstractMatrix{T}    # The stage input constraint coefficient matrix
    g::AbstractVector{T}    # The stage right hand side of the constraints
end


function ConstrainedTimeInvariantLQR( sys::AbstractStateSpace, N::Integer, Q::AbstractNumOrUniform, R::AbstractNumOrUniform, P::AbstractNumOrUniformOrSymbol = :dare;
                                      S::AbstractNumOrMatrix = 0, K::AbstractNumOrMatrixOrSymbol = 0, E::AbstractNumOrMatrix = 0, F::AbstractNumOrMatrix = 0, g::AbstractNumOrVector = 0 )
    T = promote_type( eltype( Q ), eltype( R ), eltype( F ), eltype( E ), eltype( g ), eltype( S ) )
    nx = nstates( sys )
    nu = ninputs( sys )

    # Handle the UniformScaling cases
    if( typeof( Q ) <: UniformScaling )
        Q = to_matrix( T, Q, nx )
    else
        Q = to_matrix( T, Q )
    end

    if( typeof( R ) <: UniformScaling )
        R = to_matrix( T, R, nu )
    else
        R = to_matrix( T, R )
    end


    # Verify arguments
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    if( !isposdef(R) || !issymmetric(R) )
        throw( DomainError( R, "R must be symmetric positive definite") )
    end

    if( !issymmetric(Q) )
        throw( DomainError( Q, "Q must be symmetric" ) )
    end

    if( size( Q, 1 ) != nx )
        throw( DimensionMismatch( "Q must be square with dimension the same as the number of states" ) )
    end

    if( size( R, 1 ) != nu )
        throw( DimensionMismatch( "R must be square with dimension the same as the number of rows" ) )
    end

    if( S != 0 )
        if( size( S, 1 ) != nx || size( S, 2 ) != nu )
            throw( DimensionMismatch( "S must have $nx rows and $nu columns" ) )
        end
        S = to_matrix( T, S )
    else
        S = zeros( T, nx, nu )
    end


    if( typeof( K ) <: Symbol )
        if( K == :dlqr )
            _, _, K = ared( sys.A, sys.B, R, Q )
        else
            throw(  ArgumentError( "Unknown value $K for K. Allowed values are :dlqr or a $nx by $nu matrix." ) )
        end

    elseif( K != 0 )
        if( size( K, 1 ) != nu || size( K, 2 ) != nx )
            throw( DimensionMismatch( "K must have $nx rows and $nu columns" ) )
        end
        K = to_matrix( T, K )

    else
        K = zeros( T, nu, nx )
    end

    # Create the controlled state weighting matrix (this is just the normal weighting matrix if there is no prestabilization)
    Qₖ = Q + K'*R*K


    # Handle the final state weighting matrix
    if( typeof( P ) <: Symbol )
        if( P == :Q )
            P = Qₖ

        elseif( P == :dare )
            # This computation is done using the original weighting matrices
            P, = ared( sys.A, sys.B, R, Q )

        elseif( P == :dlyap )
            # This computation is done using the original weighting matrices
            P = lyapd( sys.A', Q )

        else
            throw( ArgumentError( "Unknown value $P for P. Allowed values are :Q, :dare, :dlyap or a $nx by $nx matrix." ) )
        end

    elseif( typeof( P ) <: UniformScaling )
        P = to_matrix( T, P, nx )

    else
        if( size( P, 1 ) != nx )
            throw( DimensionMismatch( "P must be square with dimension the same as the number of states" ) )
        elseif( !issymmetric(P) )
            throw( DomainError( P, "P must be symmetric" ) )
        end

        P = to_matrix( T, P )
    end


    # Verify the constraints when there are some
    if( g != 0 )
        if( size( E, 1 ) != size( F, 1 ) )
            throw( DimensionMismatch( "E and F must have the same number of rows" ) )
        end

        if( size( E, 2 ) != nx )
            throw( DimensionMismatch( "E must have $nx columns" ) )
        end

        if( size( F, 2 ) != nu )
            throw( DimensionMismatch( "F must have $nu columns" ) )
        end

        numConstraints = size( F, 1 )
        if( size( g, 1 ) != numConstraints )
            throw( DimensionMismatch( "g must have $size( F, 1 ) rows" ) )
        end

        E = to_matrix( T, E )
        F = to_matrix( T, F )
    else
        # Create an empty matrix for the constraints
        E = zeros( T, 1, nx )
        F = zeros( T, 1, nu )
        g = zeros( T, 1 )
    end

    ConstrainedTimeInvariantLQR{T}( sys, N, Q, Qₖ, R, P, S, K, E, F, g )
end


"""
    getsystem(clqr::ConstrainedTimeInvariantLQR; prestabilized::Bool = true)

Constructs the potentially prestabilized system for a specific constrained LQR controller.

If a prestabilizing controller `K` is given in the LQR controller, that controller will be applied
to the system forming `A - BK` when `prestabilized` is true and the controlled system will be returned. If `prestabilized`
is false, the raw system is returned.
"""
function getsystem( clqr::ConstrainedTimeInvariantLQR; prestabilized::Bool = true )
    sys = clqr.sys

    if( prestabilized )
        return StateSpace( sys.A - sys.B*clqr.K, sys.B, sys.C, sys.D, sys.Ts )
    else
        return sys
    end
end


function isprestabilized( clqr::ConstrainedTimeInvariantLQR )
    return !iszero( clqr.K )
end


################################################################################
# Pretty print the CLQR struct
################################################################################
function _string_mat_with_headers(X::Matrix)
    p = (io, m) -> Base.print_matrix(io, m)
    return replace(sprint(p, X), "\"" => " ")
end


function print_constraints( io, clqr )
    rowsE = UnitRange( axes( clqr.E, 1 ) )
    colsE = UnitRange( axes( clqr.E, 2 ) )
    alignE = Base.alignment( io, clqr.E, rowsE, colsE, typemax( Int ), typemax( Int ), 2 )

    rowsF = UnitRange( axes( clqr.F, 1 ) )
    colsF = UnitRange( axes( clqr.F, 2 ) )
    alignF = Base.alignment( io, clqr.F, rowsF, colsF, typemax( Int ), typemax( Int ), 2 )

    rowsg = UnitRange( axes( clqr.g, 1 ) )
    colsg = UnitRange( axes( clqr.g, 2 ) )
    aligng = Base.alignment( io, clqr.g, rowsg, colsg, typemax( Int ), typemax( Int ), 2 )

    numConstraints = length( rowsF )

    # Figure out if a space should be added before the first elements
    v = view( clqr.E, :, 1 )
    spaceE = ( minimum( v ) >= 0 )

    v = view( clqr.F, :, 1 )
    spaceF = ( minimum( v ) >= 0 )

    v = view( clqr.g, :, 1 )
    spaceg = ( minimum( v ) >= 0 )


    # rowsF = rowsA - otherwise constraints are invalid
    for i in rowsF
        if( i == first( rowsF ) )
            premat  = "⎡"
            postmat = " ⎤"
        elseif( i == last( rowsF ) )
            premat  = "⎣"
            postmat = " ⎦"
        else
            premat  = "⎢"
            postmat = " ⎥"
        end

        print( io,  spaceE ? premat * " " : premat )
        Base.print_matrix_row( io, clqr.E, alignE, i, colsE, "  " )
        print( io, postmat )

        if( i == ceil( numConstraints / 2 ) )
            print( io, " x + " )
        else
            print( io, "     " )
        end

        print( io,  spaceF ? premat * " " : premat )
        Base.print_matrix_row( io, clqr.F, alignF, i, colsF, "  " )
        print( io, postmat )

        if( i == ceil( numConstraints / 2 ) )
            print( io, " u ≤ " )
        else
            print( io, "     " )
        end

        print( io,  spaceg ? premat * " " : premat )
        Base.print_matrix_row( io, clqr.g, aligng, i, colsg, "  " )
        print( io, postmat )

        println( io )
    end
end


Base.print(io::IO, clqr::ConstrainedTimeInvariantLQR) = show(io, clqr)


function Base.show(io::IO, clqr::ConstrainedTimeInvariantLQR)
    println( io, "Cost function:" )
    println( io, "Q = \n", _string_mat_with_headers( clqr.Q ) )
    println( io, "R = \n", _string_mat_with_headers( clqr.R ) )
    println( io, "P = \n", _string_mat_with_headers( clqr.P ) )
    if( !iszero( clqr.S ) )
        println( io, "S = \n", _string_mat_with_headers( clqr.S ) )
    end
    println( io )

    println( io, "Predicted system:" )
    print( io, clqr.sys )
    println( io )
    println( io )

    if( isprestabilized( clqr ) )
        println( io, "Prestabilizing controller:" )
        println( io, "K = \n", _string_mat_with_headers( clqr.K ) )
    else
        println( io, "No prestabilizing controller" )
    end
    println( io )

    if( iszero( clqr.E ) && iszero( clqr.F ) )
        println( "No constraints" )
    else
        println( io, "Constraints:" )
        println( io )
        print_constraints( io, clqr )
    end
end
