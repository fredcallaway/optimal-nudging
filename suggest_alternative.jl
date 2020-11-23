SUGGEST_VERSION = :best_feature

function apply_suggestion!(b::Belief, s::State, suggestion::Int)
    feature = 
        SUGGEST_VERSION == :best_feature ? argmax(s.payoffs[:, suggestion]) :
        SUGGEST_VERSION == :max_weight ? argmax(std.(b.matrix[:, suggestion])) :
        error("bad SUGGEST_VERSION")
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
    cv = choice_values(s)
    suggestion = rand(1:m.n_gamble)
    other_vals = [cv[g] for g in 1:m.n_gamble if g != suggestion]
    feature = argmax(s.payoffs[:, suggestion])
    (suggestion,
     with = simulate_suggestion(pol, s, suggestion),
     without = simulate(pol, s, Belief(s)),
     weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     mean_other = mean(other_vals),
     max_other = maximum(other_vals),
     total_val = mean(s.payoffs[:, suggestion]),
     nonbest_val = mean(s.payoffs[o, suggestion] for o in 1:m.n_outcome if o != feature))
end
