using Optim, BlackBoxOptim, NLopt, PyCall
using Optimization, OptimizationBBO, OptimizationOptimJL, OptimizationNLopt

import BlackBoxOptimizationBenchmarking: minimizer, minimum, optimize 
import Base.string

box(D) = fill((-5.5, 5.5), D)
pinit(D) = 10*rand(D).-5

# Define optimize, minimum and minimizer for each optimizer

## NLopt via OptimizationNLopt

    mutable struct NLoptOptimMethod 
        s::Symbol
    end
    string(opt::NLoptOptimMethod) = string("NLopt.", opt.s)

    function optimize(NLmeth::NLoptOptimMethod, f, D, run_length)
        opt_f = OptimizationFunction((x, p) -> f(x), Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, pinit(D), nothing, lb=-5.5*ones(D), ub=5.5*ones(D))
        
        # Create NLopt optimizer through OptimizationNLopt
        optimizer = OptimizationNLopt.NLopt.Opt(NLmeth.s, D)
        
        # Configure the optimizer
        OptimizationNLopt.NLopt.maxeval!(optimizer, run_length)
        OptimizationNLopt.NLopt.xtol_abs!(optimizer, 1e-12)
        OptimizationNLopt.NLopt.xtol_rel!(optimizer, 1e-12)
        OptimizationNLopt.NLopt.ftol_abs!(optimizer, 1e-12)
        OptimizationNLopt.NLopt.ftol_rel!(optimizer, 1e-12)
        
        sol = solve(prob, optimizer, maxiters=run_length)
        return NLmeth, sol.u, sol.objective
    end
    minimum(mfit::Tuple{NLoptOptimMethod, Vector{Float64}, Float64}) = mfit[3]
    minimizer(mfit::Tuple{NLoptOptimMethod, Vector{Float64}, Float64}) = mfit[2]

## chain

    mutable struct Chain{T, K}
        first::T
        second::K
        p::Float64
    end
    
    function optimize(m::Chain, f, D, run_length) 
        rl1 = round(Int, m.p*run_length)
        rl2 = run_length - rl1
        
        mfit = optimize(m.first, f, D, run_length)
        xinit = minimizer(mfit)
        mfit = optimize(m.second, f, D, run_length, xinit) 
    end
    
    string(m::Chain) = string(string(m.first), " → ", string(m.second))
    

## python cma

    @pyimport cma
    
    struct PyCMA
    end

    function optimize(m::PyCMA, f, D, run_length)
        # Use the Python interface directly, as we don't have an Optimization.jl wrapper for Python CMA
        es = cma.CMAEvolutionStrategy(pinit(D), 3, Dict("verb_log"=>0, "verb_disp"=>0, "maxfevals"=>run_length))
        mfit = es.optimize(f).result
        (m, mfit[1], mfit[2])
    end

    string(m::PyCMA) = "PyCMA"
    minimum(mfit::Tuple{PyCMA, Vector{Float64}, Float64}) = mfit[3]
    minimizer(mfit::Tuple{PyCMA, Vector{Float64}, Float64}) = mfit[2]
  
## scipy

    @pyimport scipy.optimize as scipy_opt

    struct PyMinimize
        method::String
    end

    # We keep the direct Python interface for scipy, as we don't have an Optimization.jl wrapper
    optimize(m::PyMinimize, f, D, run_length) = (m, scipy_opt.minimize(
        f, pinit(D), method=m.method,
        options = Dict(
            "maxfev"=>run_length, "xatol"=>1e-8, "fatol"=>1e-8,
            "maxiter"=>run_length, "gtol"=>1e-12,
        )
    ))
    minimum(mfit::Tuple{PyMinimize, Dict{Any, Any}}) = mfit[2]["fun"]
    minimizer(mfit::Tuple{PyMinimize, Dict{Any, Any}}) = mfit[2]["x"]
    
    string(m::PyMinimize) = string("Py.", m.method)
    
## Optim via OptimizationOptimJL

    function optimize(opt::Optim.AbstractOptimizer, f, D, run_length)
        opt_f = OptimizationFunction((x, p) -> f(x), Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, pinit(D), nothing)
        sol = solve(prob, opt, maxiters=run_length)
        return sol
    end
            
    function optimize(opt::Optim.AbstractOptimizer, f, D, run_length, xinit)
        opt_f = OptimizationFunction((x, p) -> f(x), Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, xinit, nothing)
        sol = solve(prob, opt, maxiters=run_length)
        return sol
    end
    
    function optimize(opt::Optim.SAMIN, f, D, run_length)
        opt_f = OptimizationFunction((x, p) -> f(x), Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, pinit(D), nothing, lb=fill(-5.5, D), ub=fill(5.5, D))
        sol = solve(prob, opt, maxiters=run_length)
        return sol
    end

    string(opt::Optim.AbstractOptimizer) = string(typeof(opt).name)

    # Optim with restart
    mutable struct OptimRestart{T}
        opt::T
    end

    function optimize(opt::OptimRestart, f, D, run_length) 
        fits = [optimize(opt.opt, f, D, round(Int, run_length/20)) for i=1:20]
        mins = [minimum(fit) for fit in fits]
        fits[argmin(mins)]
    end

    string(opt::OptimRestart) = string("Restart-", string(opt.opt))

## BlackBoxOptim via OptimizationBBO

    mutable struct BlackBoxOptimMethod 
        s::Symbol
    end
    string(opt::BlackBoxOptimMethod) = string("BBO.", opt.s)
    
    function optimize(method::BlackBoxOptimMethod, f, D, run_length)
        opt_f = OptimizationFunction((x, p) -> f(x))
        prob = OptimizationProblem(opt_f, pinit(D), nothing, lb=fill(-5.5, D), ub=fill(5.5, D))
        
        # Map the old method symbol to the new BBO_ optimizer
        # This maps old symbols like :adaptive_de_rand_1_bin to BBO_adaptive_de_rand_1_bin()
        method_name = string(method.s)
        bbo_optimizer = getproperty(OptimizationBBO, Symbol("BBO_" * method_name))()
        
        sol = solve(prob, bbo_optimizer, maxiters=run_length, TraceMode=:silent)
        return sol
    end

    # These methods work with Optimization.jl solution objects
    minimum(sol::Optimization.SciMLBase.OptimizationSolution) = sol.objective
    minimizer(sol::Optimization.SciMLBase.OptimizationSolution) = sol.u