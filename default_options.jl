using QuadGK
using Memoize
include("utils.jl")
include("meta_mdp.jl")
include("meta_greedy.jl")
include("nudging_base.jl")
# %% --------

"Expected maximum of N samples from a Normal distribution"
@memoize function emax(N::Int, d::Normal)
    mcdf(x) = cdf(d, x)^N
    lo = d.μ - 10d.σ; hi = d.μ + 10d.σ
    quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
    # We can skip the negative part because the distribution is actually truncated at zero
    # - quadgk(mcdf, lo, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, hi, atol=1e-5)[1]
end

function expected_value_of_unweighted_best(m::MetaMDP)
    # variance reduces because we take the mean
    option_value_dist = Normal(m.reward_dist.μ, m.reward_dist.σ / √m.n_outcome)
    emax(m.n_gamble, option_value_dist)
end

choose_default(s::State) = argmax(sum(s.payoffs; dims=1)[:])

function apply_default!(b::Belief, default::Int)
    v = expected_value_of_unweighted_best(b.m)
    d = mutate(b.m.reward_dist, μ=v)    
    b.matrix[:, default] = d .* b.s.weights
    b
end

function sample_default_effect(m)
    pol = MetaGreedy(m, NaN)
    s = State(m)
    b = Belief(s)
    default = choose_default(s)
    nudged_b = apply_default!(Belief(s), default )
    (default, with = evaluate(pol, s, nudged_b), without = evaluate(pol, s, b))
end
