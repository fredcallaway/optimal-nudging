using Serialization
include("nudging_base.jl")
include("cost_modification.jl")

# %% ==================== Analyze results produced by validate_job.jl ====================


results = map([3,5,10]) do n_reduce
    path = "tmp/validate_attention/$n_reduce"
    map(readdir(path)) do i
        deserialize("$path/$i")
    end
end;

R = map(results) do res
    invert(mapreduce(r->r.result, vcat, res))
end

# %% --------
N = 1000000
unbounded_rational = N \ mapreduce(+, 1:N) do i
    maximum(choice_values(exp3_state()))
end
# %% --------
function rescale(x, lo, hi)
    (x - lo) / (hi - lo)
end

X = map(R) do r
    map(r) do xs
        round(rescale(mean(xs), 150, unbounded_rational); digits=2)
        # round(sem(rescale.(xs, 150, unbounded_rational)); digits=3)
        # round(rescale(mean(xs), 150, unbounded_rational); digits=3)
    end
end |> invert

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
