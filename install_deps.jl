using Pkg
println("Installing packages...")
Pkg.add(split("Distributions StatsBase JSON SplitApplyCombine QuadGK Memoize DataStructures TypedTables CSV Parameters ProgressMeter"))
include("default_options.jl")