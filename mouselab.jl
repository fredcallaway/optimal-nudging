using Parameters
using Distributions
using Printf
import Base

const TERM = 0  # termination action
# const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
const N_FEATURE = 5
# =================== Problem =================== #

"Parameters defining a class of mouselab problems."
@with_kw struct Params
    n_gamble::Int = 7
    n_outcome::Int = 4
    reward_dist::Normal = Normal(0, 1)
    weight_alpha::Float64 = 1
    cost::Float64 = 0.01
end

exp_dist(min, max) = begin
    Normal(
        (max + min) / 2,
        0.3 * (max - min),
    )
end
exp_dist(stakes::String) = begin
    Dict(
        "high" => exp_dist(0.01, 9.99),
        "low" => exp_dist(0.01, 0.25)
    )[stakes]
end

"An individual mouselab problem (with sampled values)"
struct Problem
    prm::Params
    matrix::Matrix{Float64}
    weights::Vector{Float64}
    cost::Matrix{Float64}
end
Problem(prm::Params) = begin
    @unpack reward_dist, n_outcome, n_gamble = prm
    rs = rand(reward_dist, n_outcome * n_gamble)
    Problem(
        prm,
        reshape(rs, n_outcome, n_gamble),
        rand(Dirichlet(ones(n_outcome) * prm.weight_alpha)),
        prm.cost * ones(n_outcome, n_gamble)
        # rand(Dirichlet(dispersion * ones(n_outcome
    )
end
function Base.show(io::IO, mime::MIME"text/plain", p::Problem)
    println("Problem")
    n_row, n_col = size(p.matrix)
    println("     __", "_" ^ (7 * n_col))
    for i in 1:n_row
        @printf "%2d %% ||" p.weights[i] * 100
        for j in 1:n_col
            @printf " %1.2f |" p.matrix[i, j]
        end
        println()
    end
    # println("      --", "-" ^ (7 * n_col))
end
computations(p::Problem) = 0:prod(size(p.matrix))

"A belief about the values for a Problem.
    Currently, weights and cost are assumed to be known exactly.
"
struct Belief
    matrix::Matrix{Distribution}
    weights::Vector{Float64}
    cost::Matrix{Float64}
end
"Initial belief for a given problem."
Belief(p::Problem) = begin
    Belief(
        [p.prm.reward_dist for i in 1:p.prm.n_outcome, j in 1:p.prm.n_gamble],
        p.weights,
        p.cost
    )
end
function show_belief(b::Belief, c=0)
    println("Belief")
    ci, cj = c > 0 ? get_index(b, c) : (-1, -1)
    n_row, n_col = size(b.matrix)
    println("     __", "_" ^ (7 * n_col))
    for i in 1:n_row
        @printf "%2d %% ||" b.weights[i] * 100
        for j in 1:n_col
            d = b.matrix[i, j]
            if i == ci && j == cj
                print("  XX  |")
            elseif d.σ > 1e-10
                @printf " _%2d_ |" (j-1)*n_row + i
            else
                @printf " %1.2f |" d.μ
            end
        end
        println()
    end
    # println("      --", "-" ^ (7 * n_col))
end
Base.show(io::IO, mime::MIME"text/plain", b::Belief) = show_belief(b)
"Expected value of terminating computation with a given belief."
term_reward(b::Belief) = maximum(b.weights' * mean.(b.matrix))

"Update a belief by observing the true value of a cell."
function observe!(b::Belief, p::Problem, c::Int)
    @assert b.matrix[c].σ > 1e-10
    b.matrix[c] = Normal(p.matrix[c], 1e-20)
end

function observe(b::Belief, p::Problem, c::Int)::Belief
    b1 = deepcopy(b)
    observe!(b1, p, c)
    b1
end

"Update a belief by sampling the value of a cell."
function observe!(b::Belief, c::Int)
    @assert b.matrix[c].σ > 1e-10
    val = rand(b.matrix[c])
    # Belief.matrix contains only Normals, so we represent certainty
    # with an extremeley low variance.
    b.matrix[c] = Normal(val, 1e-20)
end

"Returns a new Belief after sampling the value of a cell"
function observe(b::Belief, c::Int)::Belief
    b1 = deepcopy(b)
    observe!(b1, c)
    b1
end

observed(b::Belief, cell::Int) = b.matrix[cell].σ == 1e-20
unobserved(b::Belief) = filter(c -> !observed(b, c), 1:length(b.matrix))

"Value of each gamble according to a belief"
function gamble_values(b::Belief)::Vector{Normal{Float64}}
    sum(b.weights .* b.matrix, dims=1)[:]
end

get_index(b::Belief, c::Int) = Tuple(CartesianIndices(size(b.matrix))[c])
get_index(p::Problem, c::Int) = Tuple(CartesianIndices(size(p.matrix))[c])

# =================== Features =================== #

"Define basic arithmetic operations on Normal distributions."
Base.:*(x::Number, n::Normal)::Normal = Normal(x * n.µ, x * n.σ)
Base.:*(n::Normal, x::Number)::Normal = Normal(x * n.µ, x * n.σ)
Base.:+(a::Normal, b::Normal)::Normal = Normal(a.μ + b.μ, √(a.σ^2 + b.σ^2))
Base.:+(a::Normal, x::Number)::Normal = Normal(a.μ + x, a.σ)
Base.:+(x::Number, a::Normal)::Normal = Normal(a.μ + x, a.σ)
Base.zero(x::Normal)::Normal = Normal(0, 1e-20)

"Highest value in x not including x[c]"
function competing_value(x::Vector{Float64}, c::Int)
    tmp = x[c]
    x[c] = -Inf
    val = maximum(x)
    x[c] = tmp
    val
end

"Expected maximum of a normal and a number."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

# Nested truncation is not supported by default.
# Truncated(d::Truncated, lower::Float64, upper::Float64) = begin
#     Truncated(d.untruncated, max(lower, d.lower), min(upper, d.upper))
# end

"Value of knowing the true value of a gamble."
function voi_gamble(b::Belief, gamble::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end
function voi_gamble(b::Belief, gamble::Int, gamble_dists, μ)
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end

"Value of knowing the values of all cells for an outcome."
function voi_outcome(b::Belief, cell::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    return voi_outcome(b, cell, μ)
end
function voi_outcome(b::Belief, outcome::Int, μ::Vector{Float64})
    # outcome = 1
    n_outcome, n_gamble = size(b.matrix)
    w = b.weights[outcome]
    gamble_dists = [Normal(μ[i], b.matrix[outcome, i].σ * w) for i in 1:n_gamble]
    samples = (rand(d, N_SAMPLE) for d in gamble_dists)
    mean(max.(samples...)) - maximum(μ)
end


"Value of knowing the value in a cell."
function voi1(b::Belief, cell::Int)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    return voi1(b, cell, μ)
end

function voi1(b::Belief, cell::Int, μ::Vector{Float64})::Float64
    n_outcome, n_gamble = size(b.matrix)
    outcome, gamble = Tuple(CartesianIndices(size(b.matrix))[cell])
    new_dist = Normal(0, 1e-20)
    for i in 1:n_outcome
        d = b.matrix[i, gamble]
        new_dist += b.weights[i] * (i == outcome ? d : d.μ)
    end
    cv = competing_value(µ, gamble)
    emax(new_dist, cv) - maximum(μ)
end

"Value of knowing everything."
function vpi(b::Belief)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    vpi(b, gamble_dists, μ)
    # mean(max.((rand(d, N_SAMPLE) for d in gamble_dists)...)) - maximum(μ)
end
function vpi(b::Belief, gamble_dists::Vector{Normal{Float64}}, μ::Vector{Float64})::Float64
    samples = (rand(d, N_SAMPLE) for d in gamble_dists)
    mean(max.(samples...)) - maximum(μ)
end

"Features for every computation in a given belief."
function features(b::Belief; skip=falses(4))
    n_outcome, n_gamble = size(b.matrix)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    voi_gambles = skip[2] ? zeros(n_gamble) :
      [voi_gamble(b, g, gamble_dists, μ) for g in 1:n_gamble]
    voi_outcomes = skip[3] ? zeros(n_outcome) :
      [voi_outcome(b, o) for o in 1:n_outcome]
    vpi_b = skip[4] ? 0. : vpi(b, gamble_dists, μ)

    function phi(cell)
        if observed(b, cell)
            return -1e10 * ones(5)
        end
        outcome, gamble = get_index(b, cell)
        return [
            -1,
            skip[1] ? 0. : voi1(b, cell, μ),
            voi_gambles[gamble],
            voi_outcomes[outcome],
            vpi_b
        ]
    end

    phis = [phi(a) for a in eachindex(b.matrix)]
    hcat(phis...)
end

# ========== Policy ========== #
"A metalevel policy that uses the BMPS features"
struct Policy
    θ::Vector{Float64}
end
"Selects a computation to perform in a given belief.
    e.g. Policy(θ)(b) -> c
"
voc(π, b) = (π.θ' * features(b; skip=π.θ[2:end] .== 0.))' .- b.cost[:]
(π::Policy)(b::Belief) = begin
    # voc = (π.θ' * features(b; skip=π.θ[2:end] .== 0.))' .- b.cost
    noise = 1e-10 * rand(length(b.matrix))
    v, c = findmax(voc(π, b) .+ noise)
    @assert isfinite(v)
    v <= 0 ? TERM : c
end

MetaGreedy() = Policy([0., 1, 0, 0, 0])


function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

choice_probs(b::Belief) = softmax(1e20 * b.weights' * mean.(b.matrix))[:]

function true_term_reward(p::Problem, b::Belief)
    choice_vals = p.weights' * p.matrix
    choice_vals * choice_probs(b)
end

# this is the one matt has changed
"Runs a Policy on a Problem."
function rollout(π::Policy, p::Problem; initial_belief=nothing, max_steps=100, callback=((b, c) -> nothing))
    b = initial_belief == nothing ? Belief(p) : initial_belief
    total_cost = 0
    computation_log = Int64[]
    for step in 1:max_steps
        c = (step == max_steps) ? TERM : π(b)
        push!(computation_log,c)
        callback(b, c)
        if c == TERM
            reward = term_reward(b) - total_cost
            choice = argmax(p.weights' * mean.(b.matrix))[2]
            actual_ev = (p.weights' * p.matrix[:,choice]) - total_cost
            return (belief=b, user_expected_reward=reward, total_cost=total_cost, n_steps=step,
            choice=choice, assistant_expected_reward = actual_ev,computation_log=computation_log)
        else
            total_cost += p.cost[c]
            observe!(b, p, c)
        end
    end
end


"Runs a Policy starting with a given belief."
function rollout(π::Policy, b::Belief, max_steps=100, belief_log=nothing)
    b = deepcopy(b)
    total_cost = 0
    computations = []
    for step in 1:max_steps
        if belief_log != nothing
            push!(belief_log, copy(b.matrix))
        end
        c = (step == max_steps) ? TERM : π(b)
        push!(computations, c)
        if c == TERM
            reward = term_reward(b) - total_cost
            return (belief=b, reward=reward, n_steps=step, computations=computations,
                    belief_log=belief_log)
        else
            total_cost += b.cost
            observe!(b, c)
        end
    end
end
