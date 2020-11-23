include("utils.jl")
include("meta_mdp.jl")
include("directed_cognition.jl")
include("meta_greedy.jl")

using StatsBase

function evaluate(pol::Policy, s::State, b=Belief(s), post_decision=nothing)
    total_p = 0.
    n_clicks = 0.
    cost = 0.
    choice = zeros(s.m.n_gamble)

    function recurse(b, p, n, pd)
        # print("\n>>> ", p, "  ")
        # display(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            if !pd  # not already applied
                recurse(post_decision(b), p, n, true)
            else
                n_clicks += p * n
                choice .+= p .* choice_probs(b)
                total_p += p
            end
        else
            opt_c = findall(softmax(1e20 * v) .!= 0)
            p /= length(opt_c)
            for c in opt_c
                cost += p * s.costs[c]
                recurse(observe(b, s, c), p, n+1, pd)
            end
        end
    end

    recurse(b, 1, 0, post_decision == nothing)
    @assert total_p ≈ 1 "total_p = $total_p"
    choice_val = only(choice_values(s) * choice)
    meta_return = choice_val - cost
    (;choice, cost, n_clicks, choice_val, meta_return)
end

function simulate(pol, s, b)
    x = rollout(pol, s, b)
    choice = sample(Weights(choice_probs(b)))
    (choice, payoff=choice_values(s)[choice], x.cost)
end

# %% --------

struct ExperimentWeights <: Distribution{Multivariate,Discrete}
    N::Int
    total::Int
end

function Base.rand(d::ExperimentWeights)
    T = d.total - d.N
    x = [0; sort!(rand(0:T, d.N-1)); T]
    diff(x) .+ 1
end

# %% --------
# @memoize function possible_weights(N; total=30)
#     filter(collect.(Iterators.product(fill(1:1:total, N)...))) do x
#         sum(x) ≈ total
#     end
# end


# %% --------
function experiment_state(m::MetaMDP)
    # @assert m.reward_dist == Normal(5, 1.75)
    w = rand(m.weight_dist)
    payoffs = max.(0, min.(10, round.(rand(m.reward_dist, (m.n_outcome, m.n_gamble)))))
    costs = m.cost * ones(m.n_outcome, m.n_gamble)
    State(m, w, payoffs, costs, w .* payoffs)
end