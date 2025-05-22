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
function to_optimization_function(f::BBOBFunction; autodiff=:none)
    # Create a wrapper function that conforms to the (x, p) signature expected by OptimizationFunction
    wrapper_fn = (x, p) -> f.f(x)
    
    # Create the OptimizationFunction with appropriate autodiff setting
    if autodiff === :none
        # Default: Use the OptimizationFunction constructor without an AD type for black-box problems
        return OptimizationFunction(wrapper_fn)
    else
        # Allow users to specify other AD backends if needed
        # Note: If `autodiff=nothing` is passed, Optimization.jl might default to AutoForwardDiff
        # but our function default is :none
        return OptimizationFunction(wrapper_fn, autodiff)
    end
end

"""
    to_optimization_problem(f::BBOBFunction, dimension::Int; 
                          x0=nothing, lb=-5.5, ub=5.5, autodiff=:none)

Create an OptimizationProblem from a BBOBFunction.

# Arguments
- `f::BBOBFunction`: The BBOB function to convert
- `dimension::Int`: The dimension of the problem
- `x0=nothing`: Initial point (if nothing, will generate a random point)
- `lb=-5.5`: Lower bound for all variables
- `ub=5.5`: Upper bound for all variables
- `autodiff=:none`: The autodiff backend to use (default: :none)

# Returns
- An OptimizationProblem ready to be solved
"""
function to_optimization_problem(f::BBOBFunction, dimension::Int; 
                               x0=nothing, lb=-5.5, ub=5.5, autodiff=:none)
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
                     max_iters=1000, n_trials=10, verbose=false, autodiff=:none)

Run a basic benchmark on a BBOBFunction using the provided optimizer.

# Arguments
- `f::BBOBFunction`: The function to benchmark
- `optimizer`: The optimizer to use (e.g., BBO_adaptive_de_rand_1_bin_radiuslimited())
- `dimension::Int`: The dimension of the problem
- `max_iters=1000`: Maximum number of iterations
- `n_trials=10`: Number of trials to run
- `verbose=false`: Whether to print progress information
- `autodiff=:none`: The autodiff backend to use (default: :none)

# Returns
- A NamedTuple containing benchmark results
"""
function run_bbob_benchmark(f::BBOBFunction, optimizer, dimension::Int; 
                         max_iters=1000, n_trials=10, verbose=false, autodiff=:none)
    success_count = 0
    objectives = Float64[]
    distances = Float64[]
    times = Float64[]
    
    # Success threshold
    Δf = 1e-6
    
    for i in 1:n_trials
        # Create a new problem for each trial (with different random starting point)
        prob = to_optimization_problem(f, dimension; autodiff=autodiff)
        
        # Use @btimed to get both the result and the time for a single solve execution
        # BenchmarkTools.jl handles warmup runs before this measurement.
        local timed_result
        try
            # Need local block for @btimed result assignment
            timed_result = @btimed solve($prob, $optimizer; maxiters=$max_iters)
        catch e
            println("Warning: Trial $i failed for optimizer on function $(f.name) in dimension $dimension. Error: $e")
            # Decide how to handle failure: skip trial? record NaN?
            # For now, let's skip and continue to next trial
            continue 
        end

        # Extract result and time
        result = timed_result.value
        time_trial = timed_result.time
        
        # Check for valid result (e.g., some solvers might return nothing or error objects on failure)
        if isnothing(result) || !hasproperty(result, :u) || !hasproperty(result, :objective)
             println("Warning: Trial $i yielded invalid result for optimizer on function $(f.name) in dimension $dimension.")
             continue
        end
        
        # Calculate distance to true minimizer
        # Ensure f.x_opt is accessible and has sufficient dimensions
        local dist_to_min
        if length(f.x_opt) >= dimension
        dist_to_min = norm(result.u - f.x_opt[1:dimension])
        else
            println("Warning: Not enough optimal points defined for function $(f.name) in dimension $dimension.")
            dist_to_min = NaN # Or handle as appropriate
        end
        
        # Check if solution is successful
        is_success = result.objective < Δf + f.f_opt
        
        if is_success
            success_count += 1
        end
        
        push!(objectives, result.objective)
        push!(distances, dist_to_min)
        push!(times, time_trial)
        
        if verbose && (i == 1 || i == n_trials || i % 10 == 0)
            # Ensure result is valid before printing
             if !isnothing(result) && hasproperty(result, :objective)
                println("Trial $i/$n_trials: objective = $(result.objective), success = $is_success, time = $time_trial")
            else
                 println("Trial $i/$n_trials: Failed or invalid result.")
             end
        end
    end
    
    # Calculate statistics without using Statistics directly
    # Check if any trials were successful before calculating stats
    n = length(objectives)
    if n == 0
        println("Warning: No successful trials completed for function $(f.name) in dimension $dimension.")
        # Return default/NaN values or handle as appropriate
        return (
            success_rate = 0.0,
            mean_objective = NaN,
            mean_distance = NaN,
            mean_time = NaN,
            objectives = Float64[],
            distances = Float64[],
            times = Float64[]
        )
    end

    sum_obj = sum(objectives)
    sum_dist = sum(filter(!isnan, distances)) # Handle potential NaNs if x_opt was missing
    sum_time = sum(times)
    
    return (
        success_rate = success_count / n_trials, # Rate over total trials attempted
        mean_objective = sum_obj / n,
        mean_distance = sum_dist / count(!isnan, distances), # Mean over valid distances
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
  - `autodiff`: Autodiff backend (default: :none)

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