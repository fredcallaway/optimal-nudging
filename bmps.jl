using Random
using Distributions
using StatsBase

# include("utils.jl")
include("voi.jl")

const G = Gumbel()

BMPSWeights = NamedTuple{(:cost, :voi1, :voi_action, :vpi),Tuple{Float64,Float64,Float64,Float64}}
"A metalevel policy that uses the BMPS features"
struct BMPSPolicy <: Policy
    m::MetaMDP
    θ::BMPSWeights
    α::Float64
end
BMPSPolicy(m::MetaMDP, θ::Vector{Float64}, α=Inf) = BMPSPolicy(m, BMPSWeights(θ), float(α))

meta_greedy(m, α=Inf) = BMPSPolicy(m, [0., 1, 0, 0], α)

"Selects a computation to perform in a given belief."
(pol::BMPSPolicy)(b::Belief) = act(pol, b)


"VOC without VPI feature"
function fast_voc(pol::BMPSPolicy, b::Belief)
    n_outcome, n_gamble = size(b.matrix)
    gamble_dists = gamble_values(b)
    μ = mean.(gamble_dists)
    voi_actions = [voi_action(b, g, gamble_dists, μ) for g in 1:n_gamble]
    θ = pol.θ

    map(computations(b)) do c
        observed(b, c) && return -Inf
        outcome, gamble = get_index(b, c)
        (-b.s.costs[c] +
         -θ.cost +
         θ.voi1 * voi1(b, c, μ) +
         θ.voi_action * voi_actions[gamble])
    end
end


function voc(pol::BMPSPolicy, b::Belief)
    fast_voc(pol, b) .+ pol.θ.vpi * vpi(b)
end


CHECK_VOC1 = false
function act(pol::BMPSPolicy, b::Belief; clever=true, check_voc1=CHECK_VOC1)
    θ = pol.θ
    voc_ = fast_voc(pol, b)

    if !clever  # computationally inefficient, but clearly correct
        voc_ .+= θ.vpi * vpi(b)
        if pol.α == Inf
            v, c = findmax(voc_)
            return (v > 0) ? c : ⊥
        else
            p = softmax(pol.α .* [0; voc_])
            return sample(0:pol.m.n_arm, Weights(p))
        end
    end

    if pol.α < Inf
        # gumbel-max trick
        voc_ .+= rand(G, length(voc_)) ./ pol.α
        voc_ .-= rand(G) / pol.α  # for term action
    else
        # break ties randomly
        voc_ .+= 1e-10 * rand(length(voc_))
    end

    # Choose candidate based on cheap voc_
    v, c = findmax(voc_)

    # Immediately return if voc is already positive.
    v > 0 && return c

    # Try putting VPI weight on VOI_action (a lower bound on VPI)
    outcome, gamble = get_index(b, c)
    v + θ.vpi * voi_action(b, gamble) > 0 && return c

    if θ.vpi > 0.
        # Try actual VPI.
        v + θ.vpi * vpi(b) > 0 && return c
    end

    if check_voc1
        v, c = findmax(voc1(b))
        v > 0 && return c
    end

    # Nope.
    return ⊥
end


struct MetaGreedy <: Policy
    m::MetaMDP
    α::Float64
end
MetaGreedy(m::MetaMDP) = MetaGreedy(m, Inf)
(pol::MetaGreedy)(b::Belief) = act(pol, b)

function voc1(b::Belief)
    μ = mean.(gamble_values(b))
    map(computations(b)) do c
        observed(b, c) && return -Inf
        voi1(b, c, μ) - b.s.costs[c]
    end
end

voc(pol::MetaGreedy, b::Belief) = voc1(b)

function act(pol::MetaGreedy, b::Belief)
    v, c = findmax(voc1(b))
    v > 0 ? c : ⊥
end




