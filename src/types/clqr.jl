
struct ConstrainedTimeInvariantLQR{T <: Number}
    """
    The predicted system
    """
    sys::AbstractStateSpace

    """
    The horizon length
    """
    N::Integer

    """
    The state weight matrix
    """
    Q::AbstractMatrix{T}

    """
    The Q weighting matrix taking into account the prestabilizing controller
    """
    Qₖ::AbstractMatrix{T}

    """
    The input weight matrix
    """
    R::AbstractMatrix{T}

    """
    The terminal state weight matrix
    """
    P::AbstractMatrix{T}

    """
    The cross-term weight matrix
    """
    S::AbstractMatrix{T}

    """
    A prestabilizing controller (if any) that forms A-BK
    """
    K::AbstractMatrix{T}

    """
    The stage state constraint coefficient matrix
    """
    E::AbstractMatrix{T}

    """
    The stage input constraint coefficient matrix
    """
    F::AbstractMatrix{T}

    """
    The stage right hand side of the constraints
    """
    g::AbstractVector{T}

    """
    The lower bounds on the state variables
    """
    xₗ::AbstractVector{Union{T, Missing}}

    """
    The upper bounds on the state variables
    """
    xᵤ::AbstractVector{Union{T, Missing}}

    """
    The lower bounds on the input variables
    """
    uₗ::AbstractVector{Union{T, Missing}}

    """
    The upper bounds on the input variables
    """
    uᵤ::AbstractVector{Union{T, Missing}}
end


function ConstrainedTimeInvariantLQR( sys::AbstractStateSpace, N::Integer, Q::AbstractNumOrUniform, R::AbstractNumOrUniform, P::AbstractNumOrUniformOrSymbol = :dare;
                                      S::AbstractNumOrMatrix = 0, K::AbstractNumOrMatrixOrSymbol = 0, E::AbstractNumOrMatrix = 0, F::AbstractNumOrMatrix = 0, g::AbstractNumOrVector = 0,
                                      xₗ::AbstractNumOrVector = -Inf, xᵤ::AbstractNumOrVector = Inf, uₗ::AbstractNumOrVector = -Inf, uᵤ::AbstractNumOrVector = Inf )
    T = promote_type( eltype( Q ), eltype( R ), eltype( F ), eltype( E ), eltype( g ), eltype( S ) )
    nₓ = nstates( sys )
    nᵤ = ninputs( sys )

    # Handle the UniformScaling cases
    if( typeof( Q ) <: UniformScaling )
        Q = to_matrix( T, Q, nₓ )
    else
        Q = to_matrix( T, Q )
    end

    if( typeof( R ) <: UniformScaling )
        R = to_matrix( T, R, nᵤ )
    else
        R = to_matrix( T, R )
    end


    # Verify arguments
    if( N < 1 )
        throw( DomainError( N, "Horizon length must be greater than 1" ) )
    end

    if( !isposdef( R ) || !issymmetric( R ) )
        throw( DomainError( R, "R must be symmetric positive definite") )
    end

    if( !issymmetric( Q ) )
        throw( DomainError( Q, "Q must be symmetric" ) )
    end

    if( size( Q, 1 ) != nₓ )
        throw( DimensionMismatch( "Q must be square with dimension the same as the number of states" ) )
    end

    if( size( R, 1 ) != nᵤ )
        throw( DimensionMismatch( "R must be square with dimension the same as the number of rows" ) )
    end

    if( S != 0 )
        if( size( S, 1 ) != nₓ || size( S, 2 ) != nᵤ )
            throw( DimensionMismatch( "S must have $nₓ rows and $nᵤ columns" ) )
        end
        S = to_matrix( T, S )
    else
        S = zeros( T, nₓ, nᵤ )
    end


    if( typeof( K ) <: Symbol )
        if( K == :dlqr )
            _, _, K = ared( sys.A, sys.B, R, Q )
        else
            throw(  ArgumentError( "Unknown value $K for K. Allowed values are :dlqr or a $nₓ by $nᵤ matrix." ) )
        end

    elseif( K != 0 )
        if( size( K, 1 ) != nᵤ || size( K, 2 ) != nₓ )
            throw( DimensionMismatch( "K must have $nₓ rows and $nᵤ columns" ) )
        end
        K = to_matrix( T, K )

    else
        K = zeros( T, nᵤ, nₓ )
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
            throw( ArgumentError( "Unknown value $P for P. Allowed values are :Q, :dare, :dlyap or a $nₓ by $nₓ matrix." ) )
        end

    elseif( typeof( P ) <: UniformScaling )
        P = to_matrix( T, P, nₓ )

    else
        if( size( P, 1 ) != nₓ )
            throw( DimensionMismatch( "P must be square with dimension the same as the number of states" ) )
        elseif( !issymmetric(P) )
            throw( DomainError( P, "P must be symmetric" ) )
        end

        P = to_matrix( T, P )
    end


    # Verify the constraints when there are some
    if( g != 0 )
        ( size( E, 1 ) != size( F, 1 ) ) && throw( DimensionMismatch( "E and F must have the same number of rows" ) )
        ( size( g, 1 ) != size( F, 1 ) ) && throw( DimensionMismatch( "g must have $size( F, 1 ) rows" ) )
        ( size( E, 2 ) != nₓ ) && throw( DimensionMismatch( "E must have $nₓ columns" ) )
        ( size( F, 2 ) != nᵤ ) && throw( DimensionMismatch( "F must have $nᵤ columns" ) )

        E = to_matrix( T, E )
        F = to_matrix( T, F )
    else
        # Create an empty matrix for the constraints
        E = zeros( T, 1, nₓ )
        F = zeros( T, 1, nᵤ )
        g = zeros( T, 1 )
    end


    # Create the bounds for the state and inputs
    ( size( xₗ, 1 ) != 1 ) && ( size( xₗ, 1 ) != nₓ ) && throw( DimensionMismatch( "xₗ must have $nₓ rows" ) )
    ( size( xᵤ, 1 ) != 1 ) && ( size( xᵤ, 1 ) != nₓ ) && throw( DimensionMismatch( "xᵤ must have $nₓ rows" ) )
    ( size( uₗ, 1 ) != 1 ) && ( size( uₗ, 1 ) != nᵤ ) && throw( DimensionMismatch( "uₗ must have $nᵤ rows" ) )
    ( size( uᵤ, 1 ) != 1 ) && ( size( uᵤ, 1 ) != nᵤ ) && throw( DimensionMismatch( "uᵤ must have $nᵤ rows" ) )

    xₗ = convertboundstomissing( T, xₗ, nₓ )
    xᵤ = convertboundstomissing( T, xᵤ, nₓ )
    uₗ = convertboundstomissing( T, uₗ, nᵤ )
    uᵤ = convertboundstomissing( T, uᵤ, nᵤ )

    ConstrainedTimeInvariantLQR{T}( sys, N, Q, Qₖ, R, P, S, K, E, F, g, xₗ, xᵤ, uₗ, uᵤ )
end

# Helper function to perform the Inf -> missing conversion on the bounds
function convertboundstomissing( T, v, n )
    if isa( v, AbstractVector )
        # Convert infinite bounds to missing
        temp = Vector{Union{T, Missing}}(undef, n)
        temp[isinf.( v )]  .= missing
        temp[.!isinf.( v )] .= v[.!isinf.( v )]

        return temp
    else
        return fill( isinf( v ) ? missing : v, (n, ) )
    end
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


function convertboundsfrommissing( T, v, n, neg )
    temp = Vector{Union{T, Float64}}( undef, n )
    temp[ismissing.( v )]  .= ( neg ? -Inf : Inf )
    temp[.!ismissing.( v )] .= v[.!ismissing.( v )]

    return temp
end

function print_bounds( io, clqr )
    nₓ = nstates( getsystem( clqr ) )
    nᵤ = ninputs( getsystem( clqr ) )
    T  = eltype( clqr.Q )

    xᵤ = convertboundsfrommissing( T, clqr.xᵤ, nₓ, false )
    xₗ = convertboundsfrommissing( T, clqr.xₗ, nₓ, true )
    uᵤ = convertboundsfrommissing( T, clqr.uᵤ, nᵤ, false )
    uₗ = convertboundsfrommissing( T, clqr.uₗ, nᵤ, true )

    rowsxᵤ  = UnitRange( axes( xᵤ, 1 ) )
    colsxᵤ  = UnitRange( axes( xᵤ, 2 ) )
    alignxᵤ = Base.alignment( io, xᵤ, rowsxᵤ, colsxᵤ, typemax( Int ), typemax( Int ), 2 )

    rowsxₗ  = UnitRange( axes( xₗ, 1 ) )
    colsxₗ  = UnitRange( axes( xₗ, 2 ) )
    alignxₗ = Base.alignment( io, xₗ, rowsxₗ, colsxₗ, typemax( Int ), typemax( Int ), 2 )

    rowsuᵤ  = UnitRange( axes( uᵤ, 1 ) )
    colsuᵤ  = UnitRange( axes( uᵤ, 2 ) )
    alignuᵤ = Base.alignment( io, uᵤ, rowsuᵤ, colsuᵤ, typemax( Int ), typemax( Int ), 2 )

    rowsuₗ  = UnitRange( axes( uₗ, 1 ) )
    colsuₗ  = UnitRange( axes( uₗ, 2 ) )
    alignuₗ = Base.alignment( io, uₗ, rowsuₗ, colsuₗ, typemax( Int ), typemax( Int ), 2 )

    # Figure out if a space should be added before the first elements
    spacexᵤ = any( xᵤ .< 0 )
    spacexₗ = any( xₗ .< 0 )
    spaceuᵤ = any( uᵤ .< 0 )
    spaceuₗ = any( uₗ .< 0 )

    anyinfxᵤ = any( isinf.( xᵤ ) )
    anyinfxₗ = any( isinf.( xₗ ) )
    anyinfuᵤ = any( isinf.( uᵤ ) )
    anyinfuₗ = any( isinf.( uₗ ) )

    havex = !all( ismissing.( clqr.xᵤ ) ) || !all( ismissing.( clqr.xₗ ) )
    haveu = !all( ismissing.( clqr.uᵤ ) ) || !all( ismissing.( clqr.uₗ ) )

    printx  = true
    printxₙ = true
    printu  = true
    printuₙ = true

    for i in max( rowsxᵤ, rowsuᵤ )
        if havex
            printx = printxₙ

            if ( i == first( rowsxᵤ ) ) && ( i == last( rowsxᵤ ) )
                printxₙ = false
                prematx  = "["
                postmatx = " ]"
            elseif i == first( rowsxᵤ )
                printxₙ = true
                prematx  = "⎡"
                postmatx = " ⎤"
            elseif  i == last( rowsxᵤ )
                printxₙ = false
                prematx  = "⎣"
                postmatx = " ⎦"
            elseif printx == true
                prematx  = "⎢"
                postmatx = " ⎥"
            else
                prematx  = " "
                postmatx = "  "
            end

            if printx == true
                print( io, anyinfxₗ ? ( isinf( xₗ[i] ) ? prematx * " " : prematx ) : prematx * " " )
                Base.print_matrix_row( io, xₗ, alignxₗ, i, colsxₗ, "  " )
                print( io, postmatx )

                if( i == ceil(  nₓ/ 2 ) )
                    print( io, " ≤ x ≤ " )
                else
                    print( io, "       " )
                end

                if spacexᵤ
                    # This case happens if there are any negative numbers in the array
                    # In that case we need to add 2 spaces before any infinities, and only
                    # 1 space in front of the negative numbers
                    print( io, isinf( xᵤ[i] ) ? prematx * "  " : prematx * " " )
                elseif anyinfxᵤ
                    # This case happens if there are infinities and only positive numbers.
                    # In that case we only add the space before the infinities.
                    print( io, isinf( xᵤ[i] ) ? prematx * " " : prematx )
                else
                    # In the case of all positive numbers, always add a space before every number
                    print( io, prematx * " " )
                end
                Base.print_matrix_row( io, xᵤ, alignxᵤ, i, colsxᵤ, "  " )
                print( io, postmatx )
            end
        end

        if havex && haveu
            print( io, "   " )
        end

        if haveu
            printu = printuₙ

            if ( i == first( rowsuᵤ ) ) && ( i == last( rowsuᵤ ) )
                printuₙ = false
                prematu  = "["
                postmatu = " ]"
            elseif i == first( rowsuᵤ )
                printuₙ = true
                prematu  = "⎡"
                postmatu = " ⎤"
            elseif i == last( rowsuᵤ )
                printuₙ = false
                prematu  = "⎣"
                postmatu = " ⎦"
            elseif printu == true
                prematu  = "⎢"
                postmatu = " ⎥"
            else
                prematu  = " "
                postmatu = "  "
            end

            if printu == true
                print( io, anyinfuₗ ? ( isinf( uₗ[i] ) ? prematu * " " : prematu ) : prematu * " " )
                Base.print_matrix_row( io, uₗ, alignuₗ, i, colsuₗ, "  " )
                print( io, postmatu )

                if( i == ceil(  nᵤ/ 2 ) )
                    print( io, " ≤ u ≤ " )
                else
                    print( io, "       " )
                end

                if spaceuᵤ
                    # This case happens if there are any negative numbers in the array
                    # In that case we need to add 2 spaces before any infinities, and only
                    # 1 space in front of the negative numbers
                    print( io, isinf( uᵤ[i] ) ? prematu * "  " : prematu * " " )
                elseif anyinfuᵤ
                    # This case happens if there are infinities and only positive numbers.
                    # In that case we only add the space before the infinities.
                    print( io, isinf( uᵤ[i] ) ? prematu * " " : prematu )
                else
                    # In the case of all positive numbers, always add a space before every number
                    print( io, prematu * " " )
                end
                Base.print_matrix_row( io, uᵤ, alignuᵤ, i, colsuᵤ, "  " )
                print( io, postmatu )
            end
        end

        println( io )
    end
end

Base.print(io::IO, clqr::ConstrainedTimeInvariantLQR) = show(io, clqr)


function Base.show(io::IO, clqr::ConstrainedTimeInvariantLQR)
    println( io, "Cost function:" )
    println( io, "Q = \n", _string_mat_with_headers( clqr.Q ) )
    println( io, "R = \n", _string_mat_with_headers( clqr.R ) )
    println( io, "P = \n", _string_mat_with_headers( clqr.P ) )

    if !iszero( clqr.S )
        println( io, "S = \n", _string_mat_with_headers( clqr.S ) )
    end
    println( io )

    println( io, "Predicted system:" )
    print( io, clqr.sys )
    println( io )
    println( io )

    if isprestabilized( clqr )
        println( io, "Prestabilizing controller:" )
        println( io, "K = \n", _string_mat_with_headers( clqr.K ) )
    else
        println( io, "No prestabilizing controller" )
    end

    println( io )

    if !all( ismissing.( clqr.xᵤ ) ) || !all( ismissing.( clqr.xₗ ) ) || !all( ismissing.( clqr.uᵤ ) ) || !all( ismissing.( clqr.uₗ ) )
        println( io, "Bounds:" )
        print_bounds( io, clqr )
    end

    if !iszero( clqr.E ) || ! iszero( clqr.F )
        println( io, "Constraints:" )
        println( io )
        print_constraints( io, clqr )
    end
end
