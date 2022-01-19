using Distributed
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV
using DataFrames
using JSON
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

M = map(Iterators.product([5], [2, 5], [2])) do (n_option, n_feature, cost)
    MetaMDP(n_option, n_feature, REWARD_DIST, ExperimentWeights(n_feature, 30), cost)
end |> collect;
@everywhere M = $M

new_effects = sample_many(sample_supersize_effect, M, 50000);
mapmany(M, new_effects) do m, de
    mapmany(de) do d
        map(d) do x
            (; mdp_features(m)..., x...)
        end
    end
end |> CSV.write("results/supersize_sims.csv")

# %% ==================== Information Highlighting ====================

@everywhere include("attention.jl")

M = map(Iterators.product([5], [3], [3], 1:28, [1.])) do (n_option, n_feature, cost, weight_highlight, α)
    m = MetaMDP(n_option, n_feature, REWARD_DIST, AttentionExperimentWeights(n_feature, 30, weight_highlight), cost)
    (m, α)
end |> collect;

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

# %% ==================== Optimal nudging ====================

@everywhere include("nudging_base.jl")
@everywhere include("optimal_nudging_base.jl")

# %% --------

function optimal_nudging_trials(n_reduce, base_cost, reduction; N=1000)
    @showprogress pmap(1:N) do i
        s, alt_costs = sample_cost_reduction_trial(;n_reduce, n_rand_reduce=n_reduce, base_cost, reduction)
        (
            problem_id = hash(s),
            payoffs = Int.(s.payoffs),
            # weights = Int.(s.weights),
            all_costs = map(x->Int.(x), alt_costs)
        )
    end
end


function simulate_nudge(m, trials)
    results = @showprogress map(trials) do t
        mapreduce(vcat, 1:10) do i
            s0 = State(m, payoffs=t.payoffs)
            map(pairs(t.all_costs)) do nudge_type, costs
                s = mutate(s0; costs)
                sim = simulate(MetaGreedy(s.m), s, Belief(s))
                (; t.problem_id, nudge_type, sim.payoff, decision_cost=sim.cost, 
                  weight_dev = sum(abs.(s.weights .- mean(s.weights))),
                  max_payoff=maximum(choice_values(s)))
            end
        end
    end
    df = reduce(vcat, results) |> DataFrame
end

# %% --------

m = MetaMDP(5, 5, REWARD_DIST, WEIGHTS(5), NaN)  # cost is overridden in simulate_nudge
trials = optimal_nudging_trials(3, 2, 2)
df = simulate_nudge(m, trials)
df |> CSV.write("results/belief_modification_sims.csv")

# %% --------

trials = optimal_nudging_trials(3, 3, 2)  # base cost 2 -> 3
#write("results/optimal_nudging_trials-3_3_2.json", json(trials))
df = simulate_nudge(m, trials)
df |> CSV.write("results/cost_reduction_sims.csv")



