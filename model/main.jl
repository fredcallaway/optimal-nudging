using Distributed
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV
using DataFrames
flatten = SplitApplyCombine.flatten
@everywhere include("nudging_base.jl")

# %% --------

function sample_many(f, M, N, args...; kws...)
    X = @showprogress pmap(Iterators.product(M, 1:N); batch_size=500) do (m, i)
        f(m, args...; kws...)
    end
    invert(splitdims(X))
end

function summarize(f, effects)
    X = map(invert(splitdims(effects))) do xs
        length(xs) .\ mapreduce(f, +, xs)
    end;
end

mdp_features(m) = (
    n_option = m.n_option,
    n_feature = m.n_feature,
    reveal_cost = m.cost
)

# %% ==================== Default options ====================
@everywhere include("default_options.jl")
mkpath("results/defaults")

M = map(Iterators.product([2, 5], [2, 5], 1:4)) do (n_option, n_feature, cost)
    MetaMDP(n_option, n_feature, REWARD_DIST, ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M

let  # pre-compute default beliefs
    db = @showprogress pmap(estimate_default_beliefs, M)
    DEFAULT_BELIEFS = Dict(zip(hash.(M), db))
    @everywhere DEFAULT_BELIEFS = $DEFAULT_BELIEFS
end

# %% --------

default_effects = sample_many(sample_default_effect, M, 10000, DCPolicy);
data = mapmany(M, default_effects) do m, de
    mapmany(de) do d
        map(0:1, [d.without, d.with]) do nudge, x
            (;mdp_features(m)...,
             nudge,
             d.weight_dev,
             x...)
        end
    end
end;
let
    # ridiculous fix for a type instability bug...
    T = typeof(data[1])
    Tdata::Vector{T} = data
    DataFrame(Tdata) |> CSV.write("results/default_sims.csv")
end


# %% ==================== Supersize ====================
@everywhere include("supersize.jl")

M = map(Iterators.product([5], [2, 5], [1,2,3,4])) do (n_option, n_feature, cost)
    MetaMDP(n_option, n_feature, REWARD_DIST, ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M

d = sample_supersize_effect(M[end])
new_effects = sample_many(sample_supersize_effect, M, 50000, DCPolicy);

data = mapmany(M, new_effects) do m, de
    mapmany(de) do d
        map(d) do x
            (; mdp_features(m)..., x...)
        end
    end
end
let
    # ridiculous fix for a type instability bug...
    T = typeof(data[1])
    Tdata::Vector{T} = data
    DataFrame(Tdata) |> CSV.write("results/supersize_sims.csv")
end
