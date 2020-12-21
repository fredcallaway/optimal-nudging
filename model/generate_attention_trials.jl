using JSON
using ProgressMeter

@everywhere include("nudging_base.jl")
@everywhere include("cost_modification.jl")

trials = @showprogress pmap(1:1000) do i
    n_reduce = 3; cost = 2
    s, alt_costs = sample_attention_trial(;n_reduce, n_rand_reduce=n_reduce, base_cost=cost, reduction=cost)
    (
        problem_id = hash(s),
        payoffs = Int.(s.payoffs),
        # weights = Int.(s.weights),
        all_costs = map(x->Int.(x), alt_costs)
    )
end

write("results/attention_trials.json", json(trials))

