using QuadGK

# =================== Utilities =================== #

"Highest value in x not including x[c]"
function competing_value(x::Vector{Float64}, c::Int)
    tmp = x[c]
    x[c] = -Inf
    val = maximum(x)
    x[c] = tmp
    val
end

"Expected maximum of distributions"
function emax(dists)
    mcdf(x) = mapreduce(*, dists) do d
        cdf(d, x)
    end

    - quadgk(mcdf, -10, 0, atol=1e-5)[1] + quadgk(x->1-mcdf(x), 0, 10, atol=1e-5)[1]
end

"Expected maximum of a distribution and and a consant."
function emax(d::Distribution, c::Float64)
    p_improve = 1 - cdf(d, c)
    p_improve < 1e-10 && return c
    (1 - p_improve)  * c + p_improve * mean(Truncated(d, c, Inf))
end
emax(x::Float64, c::Float64) = max(x, c)


# %% ==================== Features ====================

"Value of knowing the true value of a gamble."
function voi_action(b::Belief, gamble::Int,
                    gamble_dists = gamble_values(b),
                    μ = mean.(gamble_dists))
    cv = competing_value(µ, gamble)
    emax(gamble_dists[gamble], cv) - maximum(μ)
end

# function voi_outcome(b::Belief, cell::Int)
#     gamble_dists = gamble_values(b)
#     μ = mean.(gamble_dists)
#     return voi_outcome(b, cell, μ)
# end
# function voi_outcome(b::Belief, outcome::Int, μ::Vector{Float64})
#     # outcome = 1
#     n_outcome, n_gamble = size(b.matrix)
#     w = b.weights[outcome]
#     gamble_dists = [Normal(μ[i], b.matrix[outcome, i].σ * w) for i in 1:n_gamble]
#     samples = (rand(d, N_SAMPLE) for d in gamble_dists)
#     mean(max.(samples...)) - maximum(μ)
# end


"Value of knowing the value in a cell."
function voi1(b::Belief, cell::Int,
              μ = mean.(gamble_values(b)))::Float64
    observed(b, cell) && return 0.
    n_outcome, n_gamble = size(b.matrix)
    outcome, gamble = Tuple(CartesianIndices(size(b.matrix))[cell])
    new_dist = Normal(0, σ_OBS)
    for i in 1:n_outcome
        d = b.matrix[i, gamble]
        new_dist += (i == outcome ? d : d.μ)
    end
    cv = competing_value(µ, gamble)
    emax(new_dist, cv) - maximum(μ)
end

"Value of knowing everything."
function vpi(b::Belief,
             gamble_dists = gamble_values(b),
             μ = mean.(gamble_dists))
    emax(gamble_values(b)) - maximum(μ)
end

function vpi_mc(b::Belief,
             gamble_dists = gamble_values(b),
             μ = mean.(gamble_dists))
    samples = (rand(d, N_SAMPLE) for d in gamble_dists)
    mean(max.(samples...)) - maximum(μ)
end
