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
    X = @showprogress string(f) pmap(Iterators.product(M, 1:N); batch_size=500) do (m, i)
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
    db = @showprogress "estimate_default_beliefs" pmap(estimate_default_beliefs, M)
    DEFAULT_BELIEFS = Dict(zip(hash.(M), db))
    @everywhere DEFAULT_BELIEFS = $DEFAULT_BELIEFS
end

# %% --------

default_effects = sample_many(sample_default_effect, M, 10000);
mapmany(M, default_effects) do m, de
    mapmany(de) do d
        map(0:1, [d.without, d.with]) do nudge, x
            (;mdp_features(m)...,
             nudge,
             d.weight_dev,
             x...)
        end
    end
end |> CSV.write("results/default_sims.csv")

# %% ==================== Supersize ====================
@everywhere include("supersize.jl")

M = map(Iterators.product([5], [2, 5], [1,2,3,4])) do (n_option, n_feature, cost)
    MetaMDP(n_option, n_feature, REWARD_DIST, ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M

d = sample_supersize_effect(M[end])
new_effects = sample_many(sample_supersize_effect, M, 50000);

mapmany(M, new_effects) do m, de
    mapmany(de) do d
        map(d) do x
            (; mdp_features(m)..., x...)
        end
    end
end |> CSV.write("results/supersize_sims.csv")

# %% ==================== Attention ====================
@everywhere include("attention.jl")

M = map(Iterators.product([5], [3], [3], 1:28, [1.])) do (n_option, n_feature, cost, weight_highlight, α)
    m = MetaMDP(n_option, n_feature, REWARD_DIST, AttentionExperimentWeights(n_feature, 30, weight_highlight), cost)
    (m, α)
end |> collect;

d = sample_attention_effect(M[end])
attention_effects = sample_many(sample_attention_effect, M, 5000);

mapmany(M, attention_effects) do (m, α), de
    mapmany(de) do d
        map(0:1, [d.without, d.with]) do nudge, x
            (;mdp_features(m)...,
             α,
             nudge,
             d.weight_dev,
             d.weight_highlight,
             x...)
        end
    end
end |> CSV.write("results/attention_sims.csv")


# %% ==================== Attention - ALT ====================
@everywhere include("attention.jl")

M = map(Iterators.product([5], [3], [3], [1.])) do (n_option, n_feature, cost, α)
    m = MetaMDP(n_option, n_feature, REWARD_DIST, FixedWeights([2, 8, 20]), cost)
    (m, α)
end |> collect;

d = sample_attention_effect(M[end])
attention_effects = sample_many(M, 10000) do x
    sample_attention_effect(x; rand_feature=true)
end

data = mapmany(M, attention_effects) do (m, α), de
    mapmany(de) do d
        map(0:1, [d.without, d.with]) do nudge, x
            (;mdp_features(m)...,
             α,
             nudge,
             d.weight_dev,
             d.weight_highlight,
             x...)
        end
    end
end;

DataFrame(data) |> CSV.write("results/attention_sims_alt2.csv")
