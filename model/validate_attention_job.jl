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
        expected_reward(mutate(s, costs=costs))
    end
    (;rewards..., max_payoff=maximum(choice_values(s)))
end


result = map(1:5000) do j
    sample_attention_effect(;n_reduce, n_rand_reduce=n_reduce, base_cost=cost, reduction=cost)
end

serialize(out_file, (;n_reduce, cost, result))
println("Wrote $out_file")