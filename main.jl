using Distributed
@everywhere include("default_options.jl")
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV

mkpath("results/defaults")

# %% --------

function sample_many(f, M, N)
    @showprogress pmap(Iterators.product(M, 1:N)) do (m, i)
        f(m)
    end
end

function summarize(f, effects)
    X = map(invert(splitdims(effects))) do xs
        length(xs) .\ mapreduce(f, +, xs)
    end;
end


# %% ==================== Default options ====================

M = map(Iterators.product(2:2:6, 2:2:6, .02:.02:.2)) do (n_gamble, n_outcome, cost)
    MetaMDP(n_gamble, n_outcome, Normal(5, 1.5), Dirichlet(ones(n_outcome)), cost)
end |> collect;

default_effects = sample_many(sample_default_effect, M, 5000);

# %% --------
X = summarize(default_effects) do x
    [x.with.choice[x.default],  x.without.choice[x.default]]
end

map(M, X) do m, (with, without)
    (;m.n_gamble, m.n_outcome, m.cost, with, without)
end  |> (x->x[:]) |> CSV.write("results/defaults/p_choose_default.csv")

# %% ==================== Suggest alternative ====================

@everywhere include("suggest_alternative.jl")
@everywhere include("nudging_base.jl")
suggest_effects = sample_many(sample_suggestion_effect, M, 5000);

# %% --------

X = summarize(suggest_effects) do x
    [x.with.choice[x.suggestion], x.without.choice[x.suggestion]]
end

map(M, X) do m, (with, without)
    (;m.n_gamble, m.n_outcome, m.cost, with, without)
end  |> (x->x[:]) |> CSV.write("results/suggestions.csv")

# %% --------
