using ProgressMeter
@everywhere include("nudging_base.jl")
@everywhere include("cost_modification.jl")
@everywhere meta_return(s::State) = evaluate(MetaGreedy(s.m), s).meta_return

@everywhere function sample_attention_effect(;base_cost=3, reduction=2, n_reduce=5, n_rand_reduce=5, n_weight=10000)
    # fix random_select
    s = exp3_state(;base_cost, reduction, n_rand_reduce)
    alt_costs = map(x->x.costs, get_reductions(s; reduction, n_reduce))

    r = map(1:n_weight) do i
        s1 = State(s.m; payoffs=s.payoffs, costs=s.costs)
        s1.weights .+= .001 .* randn(length(s1.weights))
        map(alt_costs) do costs
            meta_return(mutate(s1, costs=costs))
        end
    end
    map(mean, invert(r))
end

results = @showprogress pmap(Iterators.product([3,5,10], 1:9600)) do (n_reduce, i)
    sample_attention_effect(;n_reduce, n_rand_reduce=n_reduce)
end

# %% --------
N = 1000000
unbounded_rational = N \ mapreduce(+, 1:N) do i
    maximum(choice_values(exp3_state()))
end

# %% --------
E = results |> splitdims |> invert .|> invert

function rescale(x, lo, hi)
    (x - lo) / (hi - lo)
end

X = map(E) do e
    map(e) do xs
        round(rescale(mean(xs), 150, unbounded_rational); digits=2)
    end
end |> invert


@show X.greedy .- X.extreme
@show X.greedy .- X.random
@show X.greedy .- X.none
@show X.none

# %% --------
