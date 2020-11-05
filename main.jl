@everywhere include("default_options.jl")
using ProgressMeter
using SplitApplyCombine
using Serialization
using CSV

mkpath("results/defaults")

# %% --------

function sample_default_effect_multi(M, N)
    @showprogress pmap(Iterators.product(M, 1:N)) do (m, i)
        sample_default_effect(m)
    end
end

M = map(Iterators.product(2:2:6, 2:2:6, .02:.02:.2)) do (n_gamble, n_outcome, cost)
    MetaMDP(n_gamble, n_outcome, Normal(5, 1.5), Dirichlet(ones(n_outcome)), cost)
end |> collect;

default_effects = sample_default_effect_multi(M, 5000);

X = map(invert(splitdims(default_effects))) do xs
    length(xs) .\ mapreduce(+, xs) do x
        [x.with.choice[x.default],  x.without.choice[x.default]]
        # [x.with.n_clicks, x.without.n_clicks]
    end
end;

map(M, X) do m, (with, without)
    (;m.n_gamble, m.n_outcome, m.cost, with, without)
end  |> (x->x[:]) |> CSV.write("results/defaults/p_choose_default.csv")
# %% --------

include("suggest_alternative.jl")


M = map(Iterators.product(2:2:6, .02:.02:.2, quality)) do (n_gamble, cost)
    MetaMDP(n_gamble, n_outcome, Normal(5, 1.5), Dirichlet(ones(n_outcome)), cost)
end |> collect;


sample_suggestion_effect(mutate(m, cost=.1), 1)