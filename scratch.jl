include("utils.jl")
include("meta_mdp.jl")
include("data.jl")
include("fitting.jl")
include("bmps.jl")
include("cost_modification.jl")
# %% ====================  ====================
t = trials[1]
State(pol.m, t.weights .* t.values, pol.m.cost .* t.costs, t.weights)

# %% ====================  ====================

using SplitApplyCombine
using Distributed
using Random
# addprocs(40)
# @everywhere cd("/usr/people/flc2/juke/cost-modification")
@everywhere include("mouselab.jl")
include("metrics.jl")
using Printf

# %% ====================  ====================
prm = Params(n_gamble=6, n_outcome=3, cost=0.02, reward_dist=Normal(0.75, 0.37), weight_alpha=5)


parse_matrix(m) = combinedims(map(x->Int.(x), (m))) ./ 100

using JSON
performance = open("trials.json") do f
    map(JSON.parse(f)) do j
        prob = Problem(prm)
        prob.matrix .= parse_matrix(j["payoffs"])
        prob.weights .= float.(j["weights"]) ./ 100
        prob.cost .= parse_matrix(j["sale_costs"])
        sale_er = expected_reward(prob)
        prob.cost .= parse_matrix(j["random_costs"])
        rand_er = expected_reward(prob)
        (problem_id=j["problem_id"],
         smart_reward=round(sale_er; digits=4),
         rand_reward=round(rand_er; digits=4))
    end
end

using CSV
performance |> CSV.write("performance.csv")

# %% ====================  ====================
juxt(fs...) = x -> Tuple(f(x) for f in fs)

function expected(f::Function, prob::Problem, pol, n=1000)
    total = @distributed (+) for i in 1:n
        f(rollout(pol, prob))
    end
    total / n
end

1
# %% ==================== Effect of noise ====================
prm = Params(n_gamble=6,n_outcome=3,cost=0.05, weight_alpha=1)
noisy = NoisyMetaGreedy(10)

x, x_noisy = map(1:50) do i
    prob = Problem(prm)
    map([meta_greedy, noisy]) do pol
        expected(prob, pol, 500) do roll
            roll.choice_value - roll.total_cost
        end
    end
end |> invert
@printf "%.2f ± %.2f\n" juxt(mean, std)(x_noisy .- x)...

# %% ==================== Std of metalevel return ====================
noisy = NoisyMetaGreedy(10)
x = map(1:50) do i
    prob = Problem(prm)
    pmap(1:1000) do i
        roll = rollout(noisy, prob)
        roll.choice_value - roll.total_cost
    end |> std
end
@printf "%.2f ± %.2f\n" mean(x) std(x)

# %% ====================  ====================
