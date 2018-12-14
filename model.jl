using Distributed
using Glob
using JLD
using DataStructures: DefaultDict
using Printf
import CSV
import JSON
using DataFrames
using DataFramesMeta

if !endswith(pwd(), "StrategyDiscovery/Journal/julia")
    cd("StrategyDiscovery/Journal/julia")
end
@everywhere include("mouselab.jl")

const JOBNAME = "hyena"

# %% ==================== Model ID ====================
#
# function name(prm::Params)
#     comp = prm.compensatory ? "" : "Non-"
#     stakes = prm.reward_dist.μ > 1 ? "High" : "Low"
#     cmult = prm.cost / 0.01
#     "$(comp)Compensatory - $stakes Stakes - $(cmult)x Cost"
# end

struct MID
    compensatory::Bool
    high_stakes::Bool
    cost::Float64
end
MID(prm::Params) = MID(
    prm.compensatory,
    prm.reward_dist.μ > 1,
    prm.cost
)
MID(p::Problem) = MID(p.prm)
Params(id::MID) = Params(
    reward_dist=exp_dist(id.high_stakes ? "high" : "low"),
    compensatory=id.compensatory,
    cost=id.cost
)

Base.string(id::MID) = @sprintf "%d-%d-%.3f" id.compensatory id.high_stakes id.cost
Base.show(io::IO, id::MID) = print(io, string(id))

# %% ==================== Load policies ====================

function load_policies()
    policies = DefaultDict{MID, Vector{Policy}}(()->[])
    for file in glob("runs/$JOBNAME/results/opt*.jld")
        result = load(file, "opt_result")
        id = MID(result[:prm])
        push!(policies[id], Policy(result[:theta]))
    end
    policies
end

const policies = load_policies()


# %% ==================== Load human data ====================

const df = CSV.read("../data/kogwis/trials.csv")

function parse_json!(df, col, T::Type)
    df[col] = map(x -> convert(T, JSON.parse(x)), df[col])
end
parse_json!(df, :clicks, Array{Int})
parse_json!(df, :ground_truth, Array{Float64})
parse_json!(df, :outcome_probs, Array{Float64})

Problem(payoffs, weights, μ, σ) = begin
    prm = Params(compensatory=maximum(weights) < 0.8, reward_dist=Normal(μ, σ))
    problem = Problem(prm, reshape(payoffs, 4, 7), weights)
end
Problem(row::DataFrameRow) = begin
    vars = [:ground_truth, :outcome_probs, :reward_mu, :reward_sigma]
    Problem(values(row[vars])...)
end

df[:problem] = map(Problem, eachrow(df))
df[:mid] = map(MID, df.problem)
for cs in df.clicks
    cs .+= 1
end

condition(id::MID) = (id.compensatory, id.high_stakes)
df[:cond] = [(id.compensatory, id.high_stakes) for id in df.mid]

struct Datum
    b::Belief
    c::Int
    cond::Tuple{Bool, Bool}
end


function parse_data(row::DataFrameRow)
    result = Datum[]
    b = Belief(row.problem)
    for c in row.clicks
        push!(result, Datum(deepcopy(b), c, row.cond))
        observe!(b, row.problem, c)
    end
    push!(result, Datum(b, 0, row.cond))
end
parse_data(df::AbstractDataFrame) = vcat(parse_data.(eachrow(df))...)
