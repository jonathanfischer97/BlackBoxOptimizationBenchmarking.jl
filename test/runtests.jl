using Test
using BlackBoxOptimizationBenchmarking, Plots, Optimization
import BlackBoxOptimizationBenchmarking.Chain
const BBOB = BlackBoxOptimizationBenchmarking

using OptimizationBBO, OptimizationOptimJL

##

map(BBOB.test_x_opt, BBOB.list_functions())

test_functions = BBOB.list_functions()

b = BBOB.benchmark(
    NelderMead(), BBOB.sphere, [100, 500, 1000], 
)

b = BBOB.benchmark(
    BenchmarkSetup(NelderMead(); isboxed=false), BBOB.sphere, [100, 500, 1000], 
)

@test length(b.success_count) == 3

b = BBOB.benchmark(
    NelderMead(), test_functions[1:3], 100:100:2000, Ntrials=10,
)

b2 = BBOB.benchmark(
    ParticleSwarm(), test_functions[1:3], 100:200:2000, Ntrials=10,
)

BBOB.compute_CI!(b, 0.1)
plot(b; title = "Benchmark", label = "NelderMead", legend = :outerright)
plot!(b2; label = "ParticleSwarm")

## OptimizationBBO

D = 2

#method = Chain(BBO_adaptive_de_rand_1_bin(), NelderMead(), 0.9)

setup = Chain(
    BenchmarkSetup(BBO_adaptive_de_rand_1_bin(), isboxed = true),
    BenchmarkSetup(NelderMead(), isboxed = false),
    0.9
)

b = BBOB.benchmark(
    setup, test_functions[1:2], 100:200:5000, Ntrials=10,
)

plot(b)

##

plot(test_functions[1])

## Test Optimization.jl Adapter

@testset "Optimization.jl Adapter" begin
    # Test function conversion
    f = test_functions[1]  # Sphere function
    dimension = 3
    
    # Test function to OptimizationFunction conversion
    opt_f1 = BBOB.to_optimization_function(f)
    @test opt_f1 isa OptimizationFunction
    @test opt_f1.adtype isa Optimization.AutoForwardDiff
    
    opt_f2 = BBOB.to_optimization_function(f; autodiff=:none)
    @test opt_f2 isa OptimizationFunction
    @test opt_f2.adtype isa Optimization.SciMLBase.NoAD
    
    # Test OptimizationProblem creation
    prob = BBOB.to_optimization_problem(f, dimension)
    @test prob isa OptimizationProblem
    @test length(prob.u0) == dimension
    @test all(prob.lb .== -5.5)
    @test all(prob.ub .== 5.5)
    
    # Test custom bounds
    prob_custom = BBOB.to_optimization_problem(f, dimension; lb=-10.0, ub=10.0)
    @test all(prob_custom.lb .== -10.0)
    @test all(prob_custom.ub .== 10.0)
    
    # Test solving with different optimizers
    optimizers = [
        BBO_adaptive_de_rand_1_bin_radiuslimited(),
        BBO_de_rand_1_bin(),
        NelderMead()
    ]
    
    for optimizer in optimizers
        result = solve(prob, optimizer; maxiters=500)
        @test result isa Optimization.SciMLBase.OptimizationSolution
        @test length(result.u) == dimension
        
        # The result should be reasonably close to the optimum
        # We use a relaxed threshold since we're just testing the interface works
        @test result.objective < f.f_opt + 1.0
    end
end

##