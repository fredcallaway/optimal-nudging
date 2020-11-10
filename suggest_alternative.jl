function apply_suggestion(b::Belief, s::State, alt::Int)
    choice_probs(b)[alt] ≈ 1 && return b  # already chosen!
    feature_σs = std.(b.matrix[:, alt])
    σ, f = findmax(feature_σs)
    σ == σ_OBS && return b  # nothing more to reveal
    i = LinearIndices(b.matrix)[f, alt]
    observe(b, s, i)
end

function make_suggest(s, alt)
    function suggest(b)
        apply_suggestion(b, s, alt)
    end
end

function sample_suggestion_effect(m)
    # TODO: reveal the highest (unweighted) feature value
    # TODO: only allow choosing among the two
    pol = MetaGreedy(m, NaN)
    s = State(m)
    b = Belief(s)
    suggestion = rand(1:m.n_gamble)
    feature = argmax(s.weights)
    (suggestion, with = evaluate(pol, s, b, make_suggest(s, suggestion)), without = evaluate(pol, s, b))
end

# function sample_suggestion_effect(m, quality)
#     pol = MetaGreedy(m, NaN)
#     s = State(m)
#     b = Belief(s)
#     suggestion = rand(1:m.n_gamble)
#     feature = argmax(s.weights)
#     μ, σ = params(m.reward_dist)
#     s.payoffs[feature, suggestion] = μ + quality * σ
#     s.weighted_payoffs[feature, suggestion] = s.weights[feature] * s.payoffs[feature, suggestion]
#     (suggestion, quality, with = evaluate(pol, s, b, make_suggest(s, suggestion)), without = evaluate(pol, s, b))
# end
