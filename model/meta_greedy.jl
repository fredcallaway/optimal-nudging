struct MetaGreedy <: Policy
    m::MetaMDP
    α::Float64
end
MetaGreedy(m::MetaMDP) = MetaGreedy(m, 1e5)
(pol::MetaGreedy)(b::Belief) = act(pol, b)

"Highest value in x not including x[c]"
function competing_value(x::Vector{Float64}, c::Int)
    tmp = x[c]
    x[c] = -Inf
    val = maximum(x)
    x[c] = tmp
    val
end

"Expected maximum of a distribution and and a consant."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)

"Value of knowing the value in a cell."
function voi1(b::Belief, cell::Int, μ=choice_values(b)[:])::Float64
    observed(b, cell) && return 0.
    n_feature, n_option = size(b.matrix)
    feature, option = Tuple(CartesianIndices(size(b.matrix))[cell])
    new_dist = Normal(0, σ_OBS)
    for i in 1:n_feature
        d = b.matrix[i, option]
        new_dist += (i == feature ? d : d.μ)
    end
    cv = competing_value(µ, option)
    emax(new_dist, cv) - maximum(μ)
end
function voc1(b::Belief)
    μ = choice_values(b)[:]
    map(computations(b)) do c
        observed(b, c) && return -Inf
        voi1(b, c, μ) - b.s.costs[c]
    end
end

voc(pol::MetaGreedy, b::Belief) = voc1(b)

function (pol::MetaGreedy)(b::Belief)
    i = sample(Weights(softmax(pol.α .* [0; voc1(b)])))
    i - 1
end
