using Serialization
include("nudging_base.jl")
include("cost_modification.jl")
meta_return(s::State) = evaluate(MetaGreedy(s.m), s).meta_return

function sample_attention_effect(;base_cost=3, reduction=2, n_reduce=5, n_rand_reduce=5, n_weight=10000)
    # fix random_select
    s = exp3_state(;base_cost, reduction, n_rand_reduce)
    @time alt_costs = map(x->x.costs, get_reductions(s; reduction, n_reduce))

    @time r = map(1:n_weight) do i
        s1 = State(s.m; payoffs=s.payoffs, costs=s.costs)
        s1.weights .+= .001 .* randn(length(s1.weights))
        map(alt_costs) do costs
            meta_return(mutate(s1, costs=costs))
        end
    end
    flush(stdout)
    map(mean, invert(r))
end

jobs = collect(Iterators.product([3,5,10], 1:1000))
n_reduce, i = jobs[parse(Int, ARGS[1])]
out_path = "tmp/validate_attention/$n_reduce"
mkpath(out_path)
out_file = "$out_path/$i"
println("Computing $out_file")

result = map(1:10) do j
    sample_attention_effect(;n_reduce, n_rand_reduce=n_reduce)
end

serialize(out_file, (;n_reduce, result))
