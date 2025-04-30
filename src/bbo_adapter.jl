# Adapter module for integrating BBOBFunction with Optimization.jl

"""
    to_optimization_function(f::BBOBFunction; autodiff=:none)

Convert a BBOBFunction to an OptimizationFunction compatible with Optimization.jl.
The function wraps the underlying function f.f into an OptimizationFunction.

# Arguments
- `f::BBOBFunction`: The BBOB function to convert
- `autodiff=:none`: The autodiff backend to use (default: :none for black-box optimization)

# Returns
- An OptimizationFunction that can be used with Optimization.jl
"""
function to_optimization_function(f::BBOBFunction; autodiff=nothing)
    # Create a wrapper function that conforms to the (x, p) signature expected by OptimizationFunction
    wrapper_fn = (x, p) -> f.f(x)
    
    # Create the OptimizationFunction with appropriate autodiff setting
    if autodiff === nothing
        return OptimizationFunction(wrapper_fn, Optimization.AutoForwardDiff())
    elseif autodiff === :none
        # Use the OptimizationFunction constructor without an AD type
        return OptimizationFunction(wrapper_fn)
    else
        return OptimizationFunction(wrapper_fn, autodiff)
    end
end

"""
    to_optimization_problem(f::BBOBFunction, dimension::Int; 
                          x0=nothing, lb=-5.5, ub=5.5, autodiff=nothing)

Create an OptimizationProblem from a BBOBFunction.

# Arguments
- `f::BBOBFunction`: The BBOB function to convert
- `dimension::Int`: The dimension of the problem
- `x0=nothing`: Initial point (if nothing, will generate a random point)
- `lb=-5.5`: Lower bound for all variables
- `ub=5.5`: Upper bound for all variables
- `autodiff=nothing`: The autodiff backend to use (if nothing, defaults to AutoForwardDiff)

# Returns
- An OptimizationProblem ready to be solved
"""
function to_optimization_problem(f::BBOBFunction, dimension::Int; 
                               x0=nothing, lb=-5.5, ub=5.5, autodiff=nothing)
    # Convert BBOBFunction to OptimizationFunction
    opt_f = to_optimization_function(f; autodiff=autodiff)
    
    # Generate initial point if not provided
    if isnothing(x0)
        x0 = 10 * rand(dimension) .- 5  # Standard initialization for BBOB
    end
    
    # Create the OptimizationProblem with bounds
    return OptimizationProblem(opt_f, x0; 
                            lb=fill(lb, dimension), 
                            ub=fill(ub, dimension))
end

"""
    run_bbob_benchmark(f::BBOBFunction, optimizer, dimension::Int; 
                     max_iters=1000, n_trials=10, verbose=false, autodiff=nothing)

Run a basic benchmark on a BBOBFunction using the provided optimizer.

# Arguments
- `f::BBOBFunction`: The function to benchmark
- `optimizer`: The optimizer to use (e.g., BBO_adaptive_de_rand_1_bin_radiuslimited())
- `dimension::Int`: The dimension of the problem
- `max_iters=1000`: Maximum number of iterations
- `n_trials=10`: Number of trials to run
- `verbose=false`: Whether to print progress information
- `autodiff=nothing`: The autodiff backend to use (if nothing, defaults to AutoForwardDiff)

# Returns
- A NamedTuple containing benchmark results
"""
function run_bbob_benchmark(f::BBOBFunction, optimizer, dimension::Int; 
                         max_iters=1000, n_trials=10, verbose=false, autodiff=nothing)
    success_count = 0
    objectives = Float64[]
    distances = Float64[]
    times = Float64[]
    
    # Success threshold
    Δf = 1e-6
    
    for i in 1:n_trials
        # Create a new problem for each trial (with different random starting point)
        prob = to_optimization_problem(f, dimension; autodiff=autodiff)
        
        # Use BenchmarkTools to measure execution time more accurately
        time_trial = @belapsed solve($prob, $optimizer; maxiters=$max_iters) samples=1 evals=1
        
        # Solve the problem (outside of timing)
        result = solve(prob, optimizer; maxiters=max_iters)
        
        # Calculate distance to true minimizer
        dist_to_min = norm(result.u - f.x_opt[1:dimension])
        
        # Check if solution is successful
        is_success = result.objective < Δf + f.f_opt
        
        if is_success
            success_count += 1
        end
        
        push!(objectives, result.objective)
        push!(distances, dist_to_min)
        push!(times, time_trial)
        
        if verbose && (i == 1 || i == n_trials || i % 10 == 0)
            println("Trial $i/$n_trials: objective = $(result.objective), success = $is_success")
        end
    end
    
    # Calculate statistics without using Statistics directly
    sum_obj = sum(objectives)
    sum_dist = sum(distances)
    sum_time = sum(times)
    n = length(objectives)
    
    return (
        success_rate = success_count / n_trials,
        mean_objective = sum_obj / n,
        mean_distance = sum_dist / n,
        mean_time = sum_time / n,
        objectives = objectives,
        distances = distances,
        times = times
    )
end

"""
    compare_optimizers(f::BBOBFunction, optimizers, dimension::Int; kwargs...)

Compare multiple optimizers on a single BBOBFunction.

# Arguments
- `f::BBOBFunction`: The function to benchmark
- `optimizers`: Vector of optimizers to compare
- `optimizer_names`: Vector of names for the optimizers
- `dimension::Int`: The dimension of the problem
- Additional keyword arguments are passed to run_bbob_benchmark, including:
  - `max_iters`: Maximum number of iterations
  - `n_trials`: Number of trials
  - `verbose`: Whether to print progress
  - `autodiff`: Autodiff backend (e.g., :none to disable)

# Returns
- A Dictionary mapping optimizer names to benchmark results
"""
function compare_optimizers(f::BBOBFunction, optimizers, optimizer_names, dimension::Int; kwargs...)
    results = Dict{String, Any}()
    
    for (i, optimizer) in enumerate(optimizers)
        name = optimizer_names[i]
        results[name] = run_bbob_benchmark(f, optimizer, dimension; kwargs...)
    end
    
    return results
end