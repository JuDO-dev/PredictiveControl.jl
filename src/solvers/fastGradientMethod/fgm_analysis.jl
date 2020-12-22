"""
    function coldstartdelta( G::Matrix{Float64}, g::Vector{Float64}; H::Matrix{Float64} = zeros( Float64, (1,1) ), L::Float64 = eigmax(H) )

Compute the Δ parameter needed when computing the upper iteration bound for a cold-started
fast gradient method.

Either the Hessian `H` or its maximum eigenvalue `L` must be passed as a parameter to the function.
"""
function coldstartdelta( G::Matrix{Float64}, g::Vector{Float64}; H::Matrix{Float64} = zeros( Float64, (1,1) ), L::Float64 = eigmax(H) )
    (n, m) = size( G )

    model = OSQP.Model()

    options = Dict( :verbose => false )

    OSQP.setup!( model; P = sparse( triu( 1.0*I(m) ) ),
                        q = zeros( Float64, (m) ),
                        A = sparse( G ),
                        u = g,
                        options... )

    results = OSQP.solve!( model )

    r = norm( results.x )
    Δ  = L/2 * r^2

    return Δ
end


"""
    upperiterationbound( ϵ::Float64, G::Matrix{Float64}, g::Vector{Float64}; H::Matrix{Float64} = = zeros( Float64, (1,1) ), L::Float64 = eigmax(H), κ::Float64 = cond(H) )

Compute the maximum number of iterations needed for the fast gradient method to reach `ϵ` sub-optimality level when projecting
into the constraint set ``G x ≦ g``.

To compute the iteration bound, either the Hessian `H` must be specified, or its largest eigenvalue `L`
and condition number `κ` must be supplied instead.
"""
function upperiterationbound( ϵ::Float64, G::Matrix{Float64}, g::Vector{Float64}; H::Matrix{Float64} = zeros( Float64, (1,1) ), L::Float64 = eigmax(H), κ::Float64 = cond(H) )
    Δ = coldstartdelta( G, g, L = L )

    a₁ = ceil( ( log(ϵ) - log(Δ) ) / ( log( 1- sqrt(1/κ) ) ) )
    a₂ = ceil( 2 * sqrt(Δ/ϵ) - 2 )

    return max( 0, min( a₁, a₂ ) )
end
