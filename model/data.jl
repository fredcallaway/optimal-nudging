using JSON
using SplitApplyCombine

struct Trial
    participant_id::Int
    problem_id::Int
    weights::Vector{Float64}
    values::Matrix{Float64}
    costs::Matrix{Float64}
    uncovered::Vector{Int}
    choice::Int
end

parse_matrix(m) = m .|> Vector{Float64} |> combinedims |> transpose |> collect

# converts indices from row-major to column-major
INVERT_INDEX = transpose(reshape(1:18, 3,6))

function Trial(d::Dict{String,Any})
    Trial(
        d["participant_id"],
        d["problem_id"],
        d["problem_weights"],
        d["problem_values"] |> parse_matrix,
        d["problem_costs"] |> parse_matrix,
        INVERT_INDEX[d["uncovered_value_vec"] .+ 1],
        # d["uncovered_value_vec"] .+ 1,
        d["choice"]
    )
end


function load_trials(file)
    map(Trial, open(JSON.parse, file))
end