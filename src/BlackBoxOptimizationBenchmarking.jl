module BlackBoxOptimizationBenchmarking

    using Distributions, Memoize, Optimization, Optim
    using LinearAlgebra, RecipesBase, BenchmarkTools

    import Optim: minimum, minimizer
    import Base: show

    export BBOBFunction, BenchmarkSetup
    
    # Optimization.jl adapter exports
    export to_optimization_function, to_optimization_problem
    export run_bbob_benchmark, compare_optimizers
    
    include("BBOBFunction.jl")
    include("benchmark.jl")
    include("plot_benchmark.jl")
    include("plot_functions.jl")
    include("bbo_adapter.jl")

end # module
