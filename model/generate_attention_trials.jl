using JSON
using Dates
using ProgressMeter

@everywhere include("nudging_base.jl")
@everywhere include("cost_modification.jl")

@everywhere function sample_attention_trial()
    s = exp3_state(n_rand_reduce=3)
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

write("results/attention_trials.json", json(trials))

# %% ====================  ====================


# %% --------
t = sample_attention_trial()

s1 = State(s.m; payoffs=s.payoffs, costs=s.costs)
s1.weights .+= .001 .* randn(length(s1.weights))
map(alt_costs) do costs
    meta_return(mutate(s1, costs=costs))
end