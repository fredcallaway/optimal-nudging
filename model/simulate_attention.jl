using ProgressMeter
using CSV
using DataFrames
flatten = SplitApplyCombine.flatten
@everywhere include("nudging_base.jl")
@everywhere include("cost_modification.jl")

# %% --------

results = @showprogress pmap(1:1000) do problem_id
    n_reduce = 5; cost = 4
    s, alt_costs = sample_attention_trial(;n_reduce, n_rand_reduce=n_reduce, base_cost=cost, reduction=cost)
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

