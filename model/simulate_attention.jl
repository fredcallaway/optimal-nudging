using ProgressMeter
using CSV
using DataFrames
@everywhere include("nudging_base.jl")
@everywhere include("cost_modification.jl")

@everywhere function simulate_attention_trial(;base_cost=3, reduction=2, n_reduce=5, n_rand_reduce=5, n_weight=10000)
    # fix random_select
    s = exp3_state(;base_cost, reduction, n_rand_reduce)
    alt_costs = map(x->x.costs, get_reductions(s; reduction, n_reduce))
    (s, alt_costs)
end


# %% --------

results = @showprogress pmap(1:1000) do problem_id
    s, alt_costs = simulate_attention_trial()
    mapmany(1:100) do i
        map(pairs(alt_costs)) do name, costs
            s1 = mutate(s, costs=costs)
            sim = simulate(MetaGreedy(s.m), s1, Belief(s1))
            (;problem_id, name, sim.payoff, decision_cost=sim.cost, 
              max_payoff=maximum(choice_values(s1)),
              weight_dev = sum(abs.(s.weights .- mean(s.weights))))
        end
    end
end


flatten(results) |> DataFrame |> CSV.write("results/attention_sims.csv")

