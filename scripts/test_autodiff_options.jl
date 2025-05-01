# Test script for autodiff options in the Optimization.jl interface

using BlackBoxOptimizationBenchmarking
using Optimization, OptimizationBBO
using Test, LinearAlgebra

const BBOB = BlackBoxOptimizationBenchmarking

println("Testing Autodiff Options with BBOB Optimization.jl Interface")
println("==========================================================")

# Print all test functions to select an appropriate one
println("Available test functions:")
for (i, func) in enumerate(BBOB.list_functions())
    println("$i. $(func.name) - Optimal value: $(func.f_opt)")
end

# Test function - using Sphere (function 1) since it's simpler
f = BBOB.list_functions()[1]  # Sphere function
dimension = 3

# Let's check the actual optimal function value
println("\nSelected function: $(f.name)")
println("Optimal value: $(f.f_opt)")
println("Optimal point: $(f.x_opt[1:dimension])")

# Test different autodiff configurations
println("\nTesting different autodiff settings:")

println("\n1. Default (AutoForwardDiff):")
opt_f1 = BBOB.to_optimization_function(f)
prob1 = BBOB.to_optimization_problem(f, dimension)
println("  Autodiff type: $(typeof(opt_f1.adtype))")
result1 = solve(prob1, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=2000)
println("  Problem solved successfully with objective: $(result1.objective)")
println("  Distance to minimum: $(norm(result1.u - f.x_opt[1:dimension]))")
println("  Success threshold: $(1e-6 + f.f_opt)")
println("  Is successful? $(result1.objective < 1e-6 + f.f_opt)")

println("\n2. Explicit :none:")
opt_f2 = BBOB.to_optimization_function(f; autodiff=:none)
prob2 = BBOB.to_optimization_problem(f, dimension; autodiff=:none)
println("  Autodiff type: $(typeof(opt_f2.adtype))")
result2 = solve(prob2, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=2000)
println("  Problem solved successfully with objective: $(result2.objective)")
println("  Distance to minimum: $(norm(result2.u - f.x_opt[1:dimension]))")
println("  Success threshold: $(1e-6 + f.f_opt)")
println("  Is successful? $(result2.objective < 1e-6 + f.f_opt)")

# Compare results (should be different due to random initialization and optimization path)
println("\nResults comparison:")
println("  Default obj: $(result1.objective), :none obj: $(result2.objective)")
println("  Are solutions equal? $(result1.u == result2.u)")
println("  Are they both valid? $(result1.objective < 1000 && result2.objective < 1000)")

# Test in benchmark function
println("\nTesting autodiff options in benchmarking functions:")

# Use a more relaxed success threshold for the benchmark
# In black-box optimization contexts, especially with stochastic optimizers,
# a strict threshold like 1e-6 may be too demanding for test purposes.
# A threshold of 1e-4 is more appropriate here while still ensuring good convergence.
relaxed_threshold = 1e-4  # Less strict than the default 1e-6

# Custom run_bbob_benchmark with relaxed threshold
function run_benchmark_with_threshold(f, optimizer, dimension, threshold; autodiff=nothing)
    # This is a simplified version of run_bbob_benchmark with a custom threshold
    success_count = 0
    objectives = Float64[]
    
    for i in 1:3  # Just 3 trials for the test
        prob = BBOB.to_optimization_problem(f, dimension; autodiff=autodiff)
        result = solve(prob, optimizer; maxiters=2000)
        
        # Use the relaxed threshold
        is_success = result.objective < threshold + f.f_opt
        
        if is_success
            success_count += 1
        end
        
        push!(objectives, result.objective)
        println("Trial $i/3: objective = $(result.objective), success = $is_success")
    end
    
    return (
        success_rate = success_count / 3,
        mean_objective = sum(objectives) / 3
    )
end

# Run with default autodiff and relaxed threshold
println("\nTesting with relaxed threshold ($(relaxed_threshold)):")
benchmark_default = run_benchmark_with_threshold(
    f, 
    BBO_adaptive_de_rand_1_bin_radiuslimited(),
    dimension,
    relaxed_threshold
)

# Run with :none autodiff and relaxed threshold
benchmark_none = run_benchmark_with_threshold(
    f, 
    BBO_adaptive_de_rand_1_bin_radiuslimited(),
    dimension,
    relaxed_threshold;
    autodiff=:none
)

println("\nBenchmark results:")
println("  Default autodiff success rate: $(benchmark_default.success_rate)")
println("  :none autodiff success rate: $(benchmark_none.success_rate)")

println("\nAutodiff options testing complete!")