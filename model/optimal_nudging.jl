using JSON
using Serialization
using ProgressMeter
using DataFrames
using CSV

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


function simulate_attention(m, trials)
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
m = MetaMDP(5, 5, REWARD_DIST, WEIGHTS(5), NaN)  # cost is overridden in simulate_attention
trials = optimal_nudging_trials(3, 2, 2)
df = simulate_attention(m, trials)
df |> CSV.write("results/belief_modification_sims.csv")

# %% --------
trials = optimal_nudging_trials(3, 3, 2)  # base cost 2 -> 3
#write("results/optimal_nudging_trials-3_3_2.json", json(trials))
df = simulate_attention(m, trials)
df |> CSV.write("results/cost_reduction_sims.csv")
