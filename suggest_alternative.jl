function apply_suggestion!(b::Belief, s::State, suggestion::Int)
    # choice_probs(b)[suggestion] ≈ 1 && return b  # already chosen!
    feature = argmax(s.payoffs[:, suggestion])
    cell = LinearIndices(b.matrix)[feature, suggestion]
    observed(b, cell) && return b  # already observed
    observe!(b, s, cell)
end

function simulate_suggestion(pol, s, suggestion)
    cv = choice_values(s)
    b = Belief(s)
    # first decision
    choice, payoff, cost = simulate(pol, s, b)
    if choice != suggestion
        apply_suggestion!(b, s, suggestion)
        # second decision
        choice, payoff, cost2 = simulate(pol, s, b)
        cost += cost2
    end
    (;choice, payoff, cost)
end

function sample_suggestion_effect(m, polclass=DCPolicy)
    # TODO: only allow choosing among the two
    pol = polclass(m)
    s = experiment_state(m)
    b = Belief(s)
    suggestion = rand(1:m.n_gamble)
    feature = argmax(s.payoffs[:, suggestion])
    (suggestion, with = simulate_suggestion(pol, s, suggestion), without = simulate(pol, s, Belief(s)),
     other_vals = mean(s.payoffs[i, suggestion] for i in 1:m.n_outcome if i != feature))
end
