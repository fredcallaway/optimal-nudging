using Distributed
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV
using DataFrames
flatten = SplitApplyCombine.flatten

# %% --------
@everywhere include("nudging_base.jl")

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

M = map(Iterators.product([2, 5], [2, 5], 1:4)) do (n_outcome, n_feature, cost)
    MetaMDP(n_outcome, n_feature, Normal(5, 1.75), ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M


# %% ==================== Default options ====================

@everywhere include("default_options.jl")
mkpath("results/defaults")
let  # pre-compute default beliefs
    db = @showprogress pmap(estimate_default_beliefs, M)
    DEFAULT_BELIEFS = Dict(zip(hash.(M), db))
    @everywhere DEFAULT_BELIEFS = $DEFAULT_BELIEFS
end

# %% --------
mdp_features(m) = (
    n_option = m.n_outcome,
    n_feature = m.n_feature,
    reveal_cost = m.cost
)
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


# %% ==================== Suggest new ====================
M = map(Iterators.product([5], [2,5], [2])) do (n_outcome, n_feature, cost)
    MetaMDP(n_outcome, n_feature, Normal(5, 1.75), ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M

@everywhere include("suggest_new.jl")
d = sample_suggest_new_effect(M[end])
new_effects = sample_many(sample_suggest_new_effect, M, 50000, DCPolicy);

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
    DataFrame(Tdata) |> CSV.write("results/suggest_new_sims.csv")
end
