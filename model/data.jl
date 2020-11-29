using JSON
using CSV
using SplitApplyCombine

struct Trial
    participant_id::Int
    nudge_type::String
    nudge_index::Int
    weights::Vector{Float64}
    payoffs::Matrix{Float64}
    costs::Matrix{Float64}
    uncovered::Vector{Int}
    choice::Int
end

parse_matrix(m) = m |> JSON.parse .|> Vector{Float64} |> combinedims |> transpose |> collect

# converts indices from row-major to column-major
invert_index(n_feature, n_option) = transpose(reshape(1:n_feature * n_option, n_feature, n_option))

function Trial(d)
    payoffs = d["payoff_matrix"] |> parse_matrix
    n_feature, n_option = size(payoffs)
    Trial(

        d["participant_id"],
        d["trial_nudge"],
        d["nudge_index"] + 1,
        d["weights"] |> JSON.parse |> Vector{Float64},
        payoffs,
        d["cost_matrix"] |> parse_matrix,
        invert_index(n_feature, n_option)[JSON.parse(d["uncovered_values"]) .+ 1],
        # d["uncovered_value_vec"] .+ 1,
        d["selected_option"] + 1
    )
end

function load_trials(name)
    map(Trial, eachrow(CSV.read("../data/$name.csv")))
end
