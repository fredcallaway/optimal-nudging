using Parameters
using Distributions
import Base
using Printf: @printf
const ⊥ = 0  # termination action
const σ_OBS = 1e-20
# const NULL_FEATURES = -1e10 * ones(4)  # features for illegal computation
const N_SAMPLE = 10000
const N_FEATURE = 5

function softmax(x)
    ex = exp.(x .- maximum(x))
    ex ./= sum(ex)
    ex
end

"Define basic arithmetic operations on Normal distributions."
Base.:*(x::Number, n::Normal)::Normal = Normal(x * n.µ, x * n.σ)
Base.:*(n::Normal, x::Number)::Normal = Normal(x * n.µ, x * n.σ)
Base.:+(a::Normal, b::Normal)::Normal = Normal(a.μ + b.μ, √(a.σ^2 + b.σ^2))
Base.:+(a::Normal, x::Number)::Normal = Normal(a.μ + x, a.σ)
Base.:+(x::Number, a::Normal)::Normal = Normal(a.μ + x, a.σ)
Base.zero(x::Normal)::Normal = Normal(0, σ_OBS)
# %% ==================== ====================

"Parameters defining a class of mouselab problems."
@with_kw struct MetaMDP
    n_gamble::Int = 7
    n_outcome::Int = 4
    reward_dist::Distribution
    weight_dist::Distribution
    cost::Float64 = 0.01
end

"An individual mouselab problem (with sampled values)"
struct State
    m::MetaMDP
    weights::Vector{Float64}
    payoffs::Matrix{Float64}
    costs::Matrix{Float64}
    weighted_payoffs::Matrix{Float64}  # weights * payoffs
end
State(m::MetaMDP, weights::Vector{Float64}, payoffs::Matrix{Float64}, costs::Matrix{Float64}) = State(
    m, weights, payoffs, costs, weights .* payoffs
)

State(m::MetaMDP) = begin
    # @unpack reward_dist, weight_dist, n_outcome, n_gamble = m
    w = rand(m.weight_dist)
    payoffs = rand(m.reward_dist, (m.n_outcome, m.n_gamble))
    costs = m.cost * ones(m.n_outcome, m.n_gamble)
    State(m, w, payoffs, costs, w .* payoffs)
end

"A belief about the values for a State."
struct Belief{T <: Distribution}
    m::MetaMDP
    s::State
    matrix::Matrix{T}
end
"Initial belief for a given problem."
Belief(s::State) = begin
    X = [s.weights[i] * s.m.reward_dist for i in 1:s.m.n_outcome, j in 1:s.m.n_gamble]
    Belief(s.m, s, X)
end
Belief(m::MetaMDP) = Belief(State(m))

computations(b::Belief) = 1:length(b.matrix)
get_weights(b::Belief) = b.s.weights

"Update a belief by observing the true value of a cell."
function observe!(b::Belief, s::State, i::Int)
    @assert !observed(b, i)
    b.matrix[i] = Normal(s.weighted_payoffs[i], σ_OBS)
end
function observe(b::Belief, s::State, c::Int)::Belief
    b1 = deepcopy(b)
    observe!(b1, s, c)
    b1
end

"Update a belief by sampling the value of a cell."
function observe!(b::Belief, i::Int)
    @assert !observed(b, i)
    val = rand(b.matrix[i])
    # Belief.weighted_payoffs contains only Normals, so we represent certainty
    # with an extremeley low variance.
    b.matrix[i] = Normal(val, σ_OBS)
end
function observe(b::Belief, c::Int)::Belief
    b1 = deepcopy(b)
    observe!(b1, c)
    b1
end

observed(b::Belief, cell::Int) = b.matrix[cell].σ == σ_OBS
unobserved(b::Belief) = filter(c -> !observed(b, c), 1:length(b.matrix))

"Value of each gamble according to a belief"
function gamble_values(b::Belief)::Vector{Normal{Float64}}
    sum(b.matrix, dims=1)[:]
end

get_index(b::Belief, c::Int) = Tuple(CartesianIndices(size(b.matrix))[c])
choice_values(s::State) = sum(s.weighted_payoffs; dims=1)
choice_values(b::Belief) = sum(mean.(b.matrix); dims=1)
choice_probs(b::Belief; α=1e20) = softmax(α * choice_values(b))[:]

"Expected value of terminating computation with a given belief."
term_reward(b::Belief) = maximum(choice_values(b))
true_term_reward(s::State, b::Belief) = first(choice_values(s) * choice_probs(b))

# %% ==================== Policy ====================
abstract type Policy end

struct RandomPolicy <: Policy
    m::MetaMDP
end
(pol::RandomPolicy)(b::Belief) = rand([0; unobserved(b)])

"Runs a Policy on a State."
function rollout(π::Policy, s::State=State(π.m), b::Belief=Belief(s); max_steps=100, callback=((b, c) -> nothing))
    @assert s.weights ≈ get_weights(b)
    reward = 0
    for step in 1:max_steps
        c = (step == max_steps) ? ⊥ : π(b)
        callback(b, c)
        if c == ⊥
            reward += term_reward(b)
            return (reward=reward, steps=step, state=s, belief=b)
        else
            reward -= s.costs[c]
            observe!(b, s, c)
        end
    end
end

function rollout(callback::Function, args...; kws...)
    rollout(args...; kws..., callback=callback)
end
# %% ==================== Extras ====================


function Base.show(io::IO, x::Normal)
    μ, σ = round.((x.μ, x.σ); digits=2)
    print(io, "Normal($μ, $σ)")
end
function Base.show(io::IO, x::Dirichlet)
    α = round.(x.alpha; digits=3)
    print(io, "Dirichlet($α)")
end

function Base.show(io::IO, m::MetaMDP)
    io = stdout
    print(io, typeof(m), "(\n")
    for fn in fieldnames(typeof(m))
        println("  $fn = ", getfield(m, fn), ",")
    end
    print(")")
end


function show_state(s::State)
    n_row, n_col = size(s.weighted_payoffs)
    println("     __", "_" ^ (8 * n_col))
    for i in 1:n_row
        @printf "%2d %% ||" s.weights[i] * 100
        for j in 1:n_col
            @printf " %5.2f |" s.weighted_payoffs[i, j]
        end
        println()
    end
    # println("      --", "-" ^ (7 * n_col))
end
Base.show(io::IO, mime::MIME"text/plain", s::State) = show_state(s)

function show_belief(b::Belief, c=0)
    ci, cj = c > 0 ? get_index(b, c) : (-1, -1)
    n_row, n_col = size(b.matrix)
    weights = get_weights(b)
    println("     __", "_" ^ (8 * n_col))
    for i in 1:n_row
        @printf "%2d %% ||" weights[i] * 100
        for j in 1:n_col
            d = b.matrix[i, j]
            if i == ci && j == cj
                print("  XX  |")
            elseif d.σ > 1e-10
                @printf " _%2d_ |" (j-1)*4 + i
            else
                @printf " %5.2f |" d.μ
            end
        end
        println()
    end
    # println("      --", "-" ^ (7 * n_col))
end
Base.show(io::IO, mime::MIME"text/plain", b::Belief) = show_belief(b)
function Base.show(io::IO, b::Belief)
    t = (sum(observed(b, c) for c in computations(b)))
    print(io, "B_$t")
end

Base.hash(m::MetaMDP, h::UInt) = reduce((a,x)-> hash(x, a), fields(m); init=hash(MetaMDP))
Base.hash(d::Dirichlet{Float64}, h::UInt) = hash(d.alpha, hash(Dirichlet, h))

