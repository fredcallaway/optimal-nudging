jobs = collect(Iterators.product([3,5,7], [2,3,4], 1:10))
n_reduce, cost, i = jobs[parse(Int, ARGS[1])]

out_path = "tmp/validate_attention/reduce-cost/$n_reduce-$cost"
mkpath(out_path)
out_file = "$out_path/$i"

if isfile(out_file)
    println("$out_file already exists!")
    exit(0)
end

println("Computing $out_file")

using Serialization
include("nudging_base.jl")
include("cost_modification.jl")

function sample_attention_effect(;kws...)
    s, alt_costs = sample_attention_trial(;kws...)
    rewards = map(alt_costs) do costs
        if KNOWN_WEIGHTS
            expected_reward(mutate(s, costs=costs))
        else
            1000 \ mapreduce(1:1000) do i
                expected_reward(State(s.m; payoffs=s.payoffs, costs=costs))
            end

    end
    (;rewards..., max_payoff=maximum(choice_values(s)))
end


result = map(1:5000) do j
    sample_attention_effect(;n_reduce, n_rand_reduce=n_reduce, base_cost=cost, reduction=cost)
end

serialize(out_file, (;n_reduce, cost, result))
println("Wrote $out_file")