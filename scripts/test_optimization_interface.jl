# Test script for the Optimization.jl interface to BlackBoxOptimizationBenchmarking.jl

using BlackBoxOptimizationBenchmarking
using Optimization, OptimizationBBO
using Test, LinearAlgebra

const BBOB = BlackBoxOptimizationBenchmarking

println("Testing BBOB Optimization.jl Interface")
println("======================================")

# Set up testing parameters
dimension = 3
max_iterations = 1000
n_trials = 5

# Get a subset of BBOB functions to test
test_functions = BBOB.list_functions()[1:5]  # Use first 5 functions

# Set up optimizers to test
optimizers = [
    BBO_adaptive_de_rand_1_bin_radiuslimited(),
    BBO_de_rand_1_bin(),
    BBO_separable_nes()
]

optimizer_names = [
    "DE Adaptive Radius Limited",
    "DE Rand 1 Bin",
    "Separable NES"
]

# Test function wrapping
println("\nTesting Adapter Functions")
println("-------------------------")

for f in test_functions
    println("Function: $(f.name)")
    
    # Test creating OptimizationFunction
    opt_f = BBOB.to_optimization_function(f)
    @assert opt_f isa OptimizationFunction "Failed to create OptimizationFunction"
    println("  ✓ Created OptimizationFunction")
    
    # Test creating OptimizationProblem
    prob = BBOB.to_optimization_problem(f, dimension)
    @assert prob isa OptimizationProblem "Failed to create OptimizationProblem"
    @assert length(prob.u0) == dimension "Problem dimension mismatch"
    println("  ✓ Created OptimizationProblem")
    
    # Test running a simple optimization
    result = solve(prob, optimizers[1]; maxiters=100)
    @assert result isa Optimization.SciMLBase.OptimizationSolution "Failed to solve problem"
    @assert length(result.u) == dimension "Solution dimension mismatch"
    println("  ✓ Solved problem successfully")
    
    println("  Final objective: $(result.objective)")
    println("  True minimum: $(f.f_opt)")
    println("  Distance to minimizer: $(norm(result.u - f.x_opt[1:dimension]))")
    println()
end

# Test benchmark function
println("\nTesting Benchmark Function")
println("-------------------------")

# Run benchmarks for each function
all_results = Dict{String, Dict{String, Any}}()

for f in test_functions
    println("Benchmarking: $(f.name)")
    
    # Compare optimizers on this function
    results = BBOB.compare_optimizers(f, optimizers, optimizer_names, dimension; 
                                   max_iters=max_iterations, 
                                   n_trials=n_trials, 
                                   verbose=false)
    
    all_results[f.name] = results
    
    # Print summary
    println("  Results:")
    for (opt_name, res) in results
        println("    $(rpad(opt_name, 25)): success_rate = $(round(res.success_rate, digits=2)), mean_objective = $(res.mean_objective), mean_distance = $(round(res.mean_distance, digits=6))")
    end
    println()
end

# Aggregate results across functions
println("\nAggregate Results")
println("----------------")

for (i, opt_name) in enumerate(optimizer_names)
    success_rates = [all_results[f.name][opt_name].success_rate for f in test_functions]
    mean_distances = [all_results[f.name][opt_name].mean_distance for f in test_functions]
    
    # Calculate means manually
    mean_success = sum(success_rates) / length(success_rates)
    mean_distance = sum(mean_distances) / length(mean_distances)
    
    println("$(rpad(opt_name, 25)): mean_success_rate = $(round(mean_success, digits=2)), mean_distance = $(round(mean_distance, digits=6))")
end

println("\nTesting complete!")