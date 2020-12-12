jobs = collect(Iterators.product([3,5,10], 1:1000))
n_reduce, i = jobs[parse(Int, ARGS[1])]
out_path = "tmp/validate_attention/$n_reduce"
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

function sample_attention_effect(;base_cost=3, reduction=2, n_reduce=5, n_rand_reduce=5, n_weight=10000)
    # fix random_select
    s = exp3_state(;base_cost, reduction, n_rand_reduce)
    @time alt_costs = map(x->x.costs, get_reductions(s; reduction, n_reduce))

    @time r = map(1:n_weight) do i
        s1 = State(s.m; payoffs=s.payoffs, costs=s.costs)
        # s1.weights .+= .001 .* randn(length(s1.weights))
        map(alt_costs) do costs
            expected_reward(mutate(s1, costs=costs))
        end
    end
    flush(stdout)
    map(mean, invert(r))
end

result = map(1:10) do j
    sample_attention_effect(;n_reduce, n_rand_reduce=n_reduce)
end

serialize(out_file, (;n_reduce, result))
println("Wrote $out_file")