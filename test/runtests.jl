using BlockArrays
using ControlSystems
using LinearAlgebra
using MatrixEquations
using PredictiveControl
using Test
using SafeTestsets

# Some utilities for testing
include( "../src/utilities.jl" )
include( "testUtils.jl" )

@testset "PredictiveControl" begin
    include( "typeTests.jl" )

    @testset "Fully condensed problem" begin
        @safetestset "Initial Propagation Matrix" begin include( "./matrices/FullCondensed_InitialPropagation.jl" ) end
        @safetestset "Prediction Matrix"          begin include( "./matrices/FullCondensed_Prediction.jl" ) end
        @safetestset "Constraint Matrix"          begin include( "./matrices/FullCondensed_Constraints.jl" ) end
    end

    @testset "Analysis" begin
        @safetestset "Condition number bound" begin include( "analysis/conditioning_bound.jl" ) end
    end

    @testset "Solvers" begin
        @testset "Iteration Utilities" begin
            @safetestset "apply" begin include( "solvers/iterationUtils/apply.jl" )    end
            @safetestset "apply" begin include( "solvers/iterationUtils/stopping.jl" ) end
        end

        @testset "Fast Gradient Method" begin
            @safetestset "FGM Algorithm"        begin include( "solvers/fastGradientMethod/fgm_algorithm.jl" ) end
            @safetestset "FGM Δ - Cold Start"   begin include( "solvers/fastGradientMethod/fgm_colddelta.jl" ) end
            @safetestset "FGM upper iter bound" begin include( "solvers/fastGradientMethod/fgm_iterationbound.jl" ) end
        end
    end
end
