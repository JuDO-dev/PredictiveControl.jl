
function createSampleLTISystem(; N::Int = 10, usedare::Bool = false, uses::Bool = false, usek::Bool = false, usee::Bool = false, usef::Bool =false )
    A = [1.0 1.0;
         0.0 1.0]
    B = [0.0;
         1.0]
    C = [1.0 0.0;
         0.0 1.0]
    D = [0.0;
         0.0]
    Ts = 0.1
    sys = StateSpace( A, B, C, D, Ts )

    Q = [1.0 1.0;
         1.0 1.0]
    R = [1.0]

    if( usedare )
        if usek
            P = :dare
        else
            P = :dlyap
        end
    else
        P = :Q
    end

    E = Array{Float64}(undef, 0, 2)
    F = Array{Float64}(undef, 0, 1)
    g = Array{Float64}(undef, 0)
    K = Array{Float64}(undef, 0, 0)

    if( uses )
      S = [1.0;
           0.0]
    else
      S = [ 0.0;
            0.0]
    end

    if( usek )
        # Define a simple controller that has both poles at 0.5
        K = [ 0.25 1.0 ]
    else
        K = [ 0.0 0.0 ]
    end

    if( usee )
        # Define a set of bound constraints between -1 <= u <= 1
        E = [ 0.0 0.0;
              0.0 0.0 ]
        F = [ 1.0;
             -1.0 ]
        g = [ g;
              1.0;
              1.0]
    end

    if( usef )
        # Define a set of bounds and an affine constraint
        E = [ E;
              1.0  0.0;
             -1.0  0.0;
              0.0  1.0;
              0.0 -1.0;
              1.0  1.0 ]
        F = [ F;
              0.0;
              0.0;
              0.0;
              0.0;
              0.0 ]
        g = [ g;
              1.0;
              1.0;
              1.0;
              1.0;
              4.0 ]
    end

    if( usef && usee )
        # Create a coupling constraint between the states and inputs
        F = [ F;
              1.0 ]
        E = [ E;
              1.0 0.0 ]
        g = [ g;
              1.0 ]
    end

    return ConstrainedTimeInvariantLQR( sys, N, Q, R, P, S=S, K=K, E=E, F=F, g=g )

end
