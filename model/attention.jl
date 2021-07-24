struct AttentionExperimentWeights <: Distribution{Multivariate,Discrete}
    N::Int
    total::Int
    weight1::Int
end

function Base.rand(d::AttentionExperimentWeights)
    rest = rand(ExperimentWeights(d.N-1, d.total - d.weight1))
    [d.weight1; rest]
end

struct FixedWeights <: Distribution{Multivariate,Discrete}
    weights::Vector{Int}
end

function Base.rand(d::FixedWeights)
    return copy(d.weights)
end

function apply_highlighting!(s::State, feature::Int)
    s.costs[feature, :] .-= 2
    s
end

function simulate_attention(pol, s, feature)
    b = Belief(s)
    choice, payoff, cost, clicks = simulate(pol, s, b)
    highlight_clicks = LinearIndices(b.matrix)[feature, :]
    highlight_values = s.payoffs[feature, :]
    highlight_value = highlight_values[choice]
    (
        n_click_highlight = isempty(clicks) ? 0 : sum(c in highlight_clicks for c in clicks),
        decision_cost = cost,
        highlight_value,
        highlight_loss = highlight_value - maximum(highlight_values),
        payoff,
    )
end

function sample_attention_effect((m, α); rand_feature=false)
    pol = MetaGreedy(m, α)
    s = experiment_state(m)
    feature = rand_feature ? rand(1:m.n_feature) : 1
    s_highlight = apply_highlighting!(deepcopy(s), feature)
    (weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     weight_highlight = s.weights[feature],
     with = simulate_attention(pol, s_highlight, feature),
     without = simulate_attention(pol, s, feature))
end
