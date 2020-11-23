
function apply_suggest_new!(b::Belief, s::State, choice::Int)
    # This is a bizarre implementation to get around the assumption of a fixed
    # payoff matrix. We use the existing belief to create a new binary choice
    # between the chosen item and a new item.

    # assign new option to the slot after the chosen one
    new = choice % b.m.n_gamble + 1
    
    # put the true values of the new item in the slot
    s.payoffs[:, new] = experiment_state(b.m).payoffs[:, new]
    s.weighted_payoffs[:, new] .= s.weights .* s.payoffs[:, new]

    # reset the belief for the new item
    b.matrix[:, new] .= s.weights .* b.m.reward_dist

    # reveal the best feature of the new item
    feature = argmax(s.payoffs[:, new])
    cell = LinearIndices(b.matrix)[feature, suggestion]
    observe!(b, s, cell)

    # make the other options terrible (effectively remove them)
    unavail = [g for g in 1:b.m.n_gamble if g ∉ (choice, new)]
    b.matrix[:, unavail] .= Normal(-10000, σ_OBS)
    new
end

function sample_suggest_new_effect(m, polclass=DCPolicy)
    # TODO: only allow choosing among the two
    pol = polclass(m)
    s = experiment_state(m)
    b = Belief(s)

    # first decision
    choice, payoff, cost = simulate(pol, s, b)
    
    new = apply_suggest_new!(b, s, choice)
    # second decision
    choice2, payoff, cost2 = simulate(pol, s, b)
    @assert choice2 == choice || choice2 == new
    cost += cost2
    # TODO: include clicked suggestion
    # TODO: try maximal feature weight
    (choose_suggested = choice2 == new, payoff, decision_cost=cost,
     weight_dev = sum(abs.(s.weights .- mean(s.weights))))
end
