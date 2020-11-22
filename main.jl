using Distributed
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV
using DataFrames
flatten = SplitApplyCombine.flatten

# %% --------
include("meta_mdp.jl")

function sample_many(f, M, N, args...; kws...)
    X = @showprogress pmap(Iterators.product(M, 1:N); batch_size=100) do (m, i)
        f(m, args...; kws...)
    end
    invert(splitdims(X))
end

function summarize(f, effects)
    X = map(invert(splitdims(effects))) do xs
        length(xs) .\ mapreduce(f, +, xs)
    end;
end

M = map(Iterators.product([2, 5], [2, 5], 1:3)) do (n_gamble, n_outcome, cost)
    MetaMDP(n_gamble, n_outcome, Normal(5, 1.75), ExperimentWeights(n_outcome, 30), cost)
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

default_effects = sample_many(sample_default_effect, M, 5000, DCPolicy);
data = mapmany(M, default_effects) do m, de
    mapmany(de) do d
        map(0:1, [d.without, d.with]) do nudge, x
            (n_option = m.n_gamble,
             n_feature = m.n_outcome,
             reveal_cost = m.cost,
             nudge,
             d.weight_var,
             d.weight_dev,
             x.payoff,
             decision_cost = x.cost,
             choose_default = Int(x.choice == d.default))
        end
    end
end;
let
    # ridiculous type instability bug...
    T = typeof(data[1])
    Tdata::Vector{T} = data
    DataFrame(Tdata) |> CSV.write("results/default_sims.csv")
end


# %% ==================== Suggest alternative ====================

@everywhere include("suggest_alternative.jl")
suggest_effects = sample_many(sample_suggestion_effect, M, 5000, DCPolicy);

mapmany(M, suggest_effects) do m, de
    # cond = (;m.n_gamble, m.n_outcome, m.cost)
    mapmany(de) do d
        # .02 < m.cost < .2 && return []
        map(0:1, [d.without, d.with]) do nudge, x
            (n_option = m.n_gamble,
             n_feature = m.n_outcome,
             reveal_cost = m.cost,
             nudge,
             x.payoff,
             decision_cost = x.cost,
             choose_suggested = Int(x.choice == d.suggestion))
        end
    end
end |> DataFrame |> CSV.write("results/suggest_sims.csv")
