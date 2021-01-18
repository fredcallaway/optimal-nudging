using JSON
using Serialization
using ProgressMeter
using DataFrames
using CSV

@everywhere include("nudging_base.jl")
@everywhere include("optimal_nudging_base.jl")

function write_attention_trials(n_reduce, base_cost, reduction; N=1000)
    trials = @showprogress pmap(1:N) do i
        s, alt_costs = sample_cost_reduction_trial(;n_reduce, n_rand_reduce=n_reduce, base_cost, reduction)
        (
            problem_id = hash(s),
            payoffs = Int.(s.payoffs),
            # weights = Int.(s.weights),
            all_costs = map(x->Int.(x), alt_costs)
        )
    end

    write("results/attention_trials-$n_reduce-$base_cost-$reduction.json", json(trials))
    serialize("tmp/attention_trials-$n_reduce-$base_cost-$reduction", trials)
    return trials
end


function simulate_attention(n_reduce, base_cost, reduction)
    m = exp3_state(;base_cost, reduction).m
    trials = deserialize("tmp/attention_trials-$n_reduce-$base_cost-$reduction")
    results = @showprogress map(trials) do t
        mapreduce(vcat, 1:10) do i
            map(pairs(t.all_costs)) do nudge_type, costs
                s = State(m, payoffs=t.payoffs, costs=costs)
                sim = simulate(MetaGreedy(s.m), s, Belief(s))
                (; t.problem_id, nudge_type, sim.payoff, decision_cost=sim.cost, 
                  max_payoff=maximum(choice_values(s)))
            end
        end
    end
    df = reduce(vcat, results) |> DataFrame
    df |> CSV.write("results/attention_sims-$n_reduce-$base_cost-$reduction.csv")
    df
end


# write_attention_trials(3, 2, 2)
# simulate_attention(3, 2, 2)

# %% --------

trials3 = write_attention_trials(3, 1, -2)
df3 = simulate_attention(3, 1, -2)

trials0 = write_attention_trials(0, 1, -2)
df0 = simulate_attention(0, 1, -2)

# %% --------
# trials = deserialize("tmp/attention_trials-0-1--2")  
map(trials3) do t
    sum(t.all_costs.greedy .≠ 1) == 6
end |> mean
# %% --------
map(trials3) do t
# %% --------
map([df0, df3]) do df
    df = df3
    df.net_payoff = df.payoff - df.decision_cost
    x = by(df, :nudge_type, :decision_cost=>mean, :payoff=>mean, :net_payoff=>mean)
    d1 = x.decision_cost_mean[1] - x.decision_cost_mean[2]
    d2 = x.payoff_mean[1] - x.payoff_mean[2]
    d1, d2, d2 - d1
end


# %% --------
head(df)
by(df0, :nudge_type, :decision_cost=>mean)
by(df0, :nudge_type, :payoff=>mean)
by(df3, :nudge_type, :decision_cost=>mean)
by(df3, :nudge_type, :payoff=>mean)



