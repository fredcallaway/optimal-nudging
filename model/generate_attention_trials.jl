using JSON
using Dates
using ProgressMeter

@everywhere include("cost_modification.jl")

@everywhere function sample_attention_trial()
    s = exp3_state()r
    alt_costs = map(get_reductions(s; n_reduce=3)) do s1
        Int.(s1.costs)
    end
    (
        problem_id=rand(1:1000000),
        payoffs=Int.(s.payoffs),
        all_costs=alt_costs,
        # weights=Int.(s.weights)
    ) 
end

trials = @showprogress pmap(1:1000) do i
    sample_attention_trial()
end

write("results/attention/attention_trials.json", json(trials))


# %% ====================  ====================
r1, r2 = map(results) do r
    s.matrix .= r.payoffs
    s.costs .= r.original_costs
    s.weights .= r.weights
    r1 = expected_reward(s)
    s.costs .= r.sale_costs
    r2 = expected_reward(s)
    (r1, r2)
end |> invert

# using HypothesisTests
# using RCall
# R"""

# ""

# # %% ====================  ====================
# display("---")
# m = Params(n_gamble=6, n_outcome=3, cost=0.08,
#              reward_dist=Normal(3.00, 1.50), weight_alpha=5)

# budget = m.cost * 3
# s = State(m)
# s.costs .+= rand(-m.cost:.01:m.cost, size(s.costs))
# sort!(s.weights; rev=true)

# show_cost(s)
# describe_rollout(policy, s)
# # println(join(collect(0:3:15) .+ argmax(s.weights), " "))
# # println(join(sortperm(s.costs[:]), " "))

# x = greedy_select(s, budget, policy)
# describe_rollout(policy, sale(s, x, budget))
# # show_cost(sale(s, best_cell_select(s), budget))

# # %% ====================  ====================
# # include("evolution.jl")
# s = State(m)
# s.costs .+= rand(-m.cost:.01:m.cost, size(s.costs))

# objective = make_objective(s, budget, policy)
# evolved = evolve_sale(s, budget, policy; pop_size=400, verbosity=10, p_mutate=0.5)
# greedy = greedy_select(s, budget, policy)

# println(objective(evolved[1]))
# println(objective(greedy))
# # %% ====================  ====================
# function JSON.lower(s::State)
#     (payoffs=round.(s.matrix; digits=2),
#      costs=s.costs,
#      weights=round.(s.weights; digits=2))
# end

# problems = [State(m), State(m)]
# # isdir("trials") || mkdir("trials")
# open("trials.json", "w+") do f
#     write(f, json(problems))
# end
