using PredictiveControl
using Test


H = [ 10.0 0.0;
       0.0 2.0 ];

b = [ 2.0;
      4.0 ];

# Run for only 40 iterations
x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, stopconditions = nothing, maxiter = 40 )
@test iter == 40

# Run until converged
x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, stopconditions = [FGM.Conjugate(1e-8)] )
@test isapprox( x, [-0.2, -2.0], atol = 1e-4 )

x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, stopconditions = [FGM.Gradient(1e-8)] )
@test isapprox( x, [-0.2, -2.0], atol = 1e-4 )

x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, stopconditions = [FGM.Best(1e-8)] )
@test isapprox( x, [-0.2, -2.0], atol = 1e-4 )


# Start on the actual point - should converge very fast
x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, x₀ = [-0.2; -2.0], stopconditions = [FGM.Best(1e-8)] )
@test iter <= 2
@test isapprox( x, [-0.2, -2.0], atol = 1e-4 )


# Test a clipped region (project onto constraints)
x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> max.(-1.0, min.(1.0, x)), stopconditions = [FGM.Best(1e-8)] )
@test isapprox( x, [-0.2, -1.0], atol = 1e-4 )


# Test the variable steping
x, iter = PredictiveControl.FastGradientMethod.fastgradientmethod( H, b, x -> x, step=FGM.VariableStep(), stopconditions = [FGM.Best(1e-8)] )
@test isapprox( x, [-0.2, -2.0], atol = 1e-4 )
