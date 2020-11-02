include("utils.jl")
include("meta_mdp.jl")
include("data.jl")
include("fitting.jl")
include("bmps.jl")

# all_trials = load_trials("matt_data.json")
all_trials = load_trials("pilot_test_data.json")
# filter!(all_trials) do t
#     !isempty(t.uncovered)
# end

# %% ====================  ====================
function meta_greedy(cost::Float64)
    m = MetaMDP(6, 3, Normal(5, 1.75), Dirichlet(ones(3)), cost)
    MetaGreedy(m, NaN)
end

function cost_mle(trials)
    res = optimize(0, 10) do cost
        -softmax_mle(meta_greedy(cost), trials).logp
    end
    cost = res.minimizer
    (cost=cost, softmax_mle(meta_greedy(cost), trials)...)
end

function rand_logp(trials)
    V, cs = build_voc_table(meta_greedy(0.), trials);
    logp(V, cs, 1e-30)
end

mle = cost_mle(all_trials)
rand_logp(all_trials)


# %% ====================  ====================

map_participants(f) = map(f, group(t->t.participant_id, all_trials))
# ind_mles = map_participants() do trials
#     cost_mle(trials)
# end

# rands = map_participants(rand_logp)

# @. getfield(ind_mles, :logp) - rands

# histogram(collect(getfield.(ind_mles, :α)), xlabel="inverse temperature")

X = map_participants() do trials
    (rand_logp=rand_logp(trials), cost_mle(trials)...)
end

using TypedTables
using CSV
Table(X) |> CSV.write("individual_fits.csv")

# %% ====================  ====================
n_click = map_participants() do trials
    map(trials) do t
        length(t.uncovered)
    end |> mean
end |> collect



# %% ====================  ====================
logps = map(0:0.05:5) do cost
    -softmax_mle(meta_greedy(cost), trials).logp
end


# %% ====================  ====================
include("data.jl")
all_trials = load_trials("pilot_test_data.json")
filter!(all_trials) do t
    !isempty(t.uncovered)
end
t = all_trials[2]
s = State(pol.m, t.weights .* t.values, pol.m.cost .* t.costs, t.weights)
t.uncovered

# %% ====================  ====================
d = filter(open(JSON.parse, "pilot_test_data.json")) do d
    d["problem_id"] == t.problem_id && d["participant_id"] == t.participant_id
end |> first

INVERT_INDEX[d["uncovered_value_vec"] .+ 1]