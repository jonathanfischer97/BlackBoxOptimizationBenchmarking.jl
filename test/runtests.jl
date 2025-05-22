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
    
    # Define an AD backend for testing
    ad_backend = Optimization.AutoForwardDiff() 

    # Test 1: Explicitly :none autodiff (NoAD)
    opt_f_none = BBOB.to_optimization_function(f; autodiff=:none)
    @test opt_f_none isa OptimizationFunction
    @test opt_f_none.adtype isa Optimization.SciMLBase.NoAD 
    
    # Test 2: Explicitly specified AD backend (e.g., AutoForwardDiff)
    opt_f_ad = BBOB.to_optimization_function(f; autodiff=ad_backend)
    @test opt_f_ad isa OptimizationFunction
    @test opt_f_ad.adtype isa Optimization.AutoForwardDiff
    
    # Test OptimizationProblem creation
    
    # Test 3: Problem with default autodiff (should be :none / NoAD)
    prob = BBOB.to_optimization_problem(f, dimension)
    @test prob isa OptimizationProblem
    @test length(prob.u0) == dimension
    @test all(prob.lb .== -5.5)
    @test all(prob.ub .== 5.5)
    @test prob.f.adtype isa Optimization.SciMLBase.NoAD # Verify default AD type

    # Test 4: Problem with explicit AD
    prob_ad = BBOB.to_optimization_problem(f, dimension; autodiff=ad_backend)
    @test prob_ad isa OptimizationProblem
    @test prob_ad.f.adtype isa Optimization.AutoForwardDiff # Verify explicit AD type
    
    # Test 5: Problem with custom bounds (using default :none)
    prob_custom = BBOB.to_optimization_problem(f, dimension; lb=-10.0, ub=10.0)
    @test all(prob_custom.lb .== -10.0)
    @test all(prob_custom.ub .== 10.0)
    @test prob_custom.f.adtype isa Optimization.SciMLBase.NoAD # Check AD type
    
    # Define optimizers
    # Separate BBO optimizers (should work with NoAD) from NelderMead (needs AD function for bounds)
    bbo_optimizers = [
        BBO_adaptive_de_rand_1_bin_radiuslimited(),
        BBO_de_rand_1_bin()
    ]
    # Optim.jl optimizers (like NelderMead) seem to need AD func handle for Fminbox
    # Note: If adding gradient-based optimizers later (e.g., LBFGS), they'd also go here.
    other_optimizers = [
        NelderMead()
    ]
    
    # Test solving with the :none problem (using BBO optimizers)
    println("Testing solve with autodiff=:none (BBO optimizers)...")
    for optimizer in bbo_optimizers
        result = solve(prob, optimizer; maxiters=500)
        @test result isa Optimization.SciMLBase.OptimizationSolution
        @test length(result.u) == dimension
        @test result.objective < f.f_opt + 1.0
    end

    # Test solving with the AD problem (using all optimizers)
    # BBO should ignore the gradient info, NelderMead requires the function handle for Fminbox
    println("Testing solve with autodiff=AutoForwardDiff (all optimizers)...")
    for optimizer in [bbo_optimizers..., other_optimizers...] # Combine lists
         try
            result_ad = solve(prob_ad, optimizer; maxiters=500)
            @test result_ad isa Optimization.SciMLBase.OptimizationSolution
            @test length(result_ad.u) == dimension
            @test result_ad.objective < f.f_opt + 1.0 
        catch e
            # Added optimizer type to the message for clarity
            println("Note: Solving AD problem failed for optimizer $(typeof(optimizer)) (might be expected): $e")
            @test false # Let's make unexpected failures explicit errors
         end
    end
end

##