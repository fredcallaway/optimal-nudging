
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

function reveal_best_outcome!(b::Belief, s::State, g::Int)
    o = argmax(s.payoffs[:, g])
    cell = LinearIndices(b.matrix)[o, g]
    observe!(b, s, cell)
end

function add_new_option(s::State, b::Belief)
    # TODO: maybe prevent clicking the other ones
    m1 = mutate(s.m, n_gamble=s.m.n_gamble+1)

    new_payoffs = round_payoffs!(rand(m1.reward_dist, m1.n_outcome))
    payoffs = [s.payoffs new_payoffs]
    s1 = State(m1, s.weights, payoffs, [s.costs m1.cost .* ones(m1.n_outcome)])

    new_beliefs = s.weights .* m1.reward_dist
    b1 = Belief(m1, s1, [b.matrix new_beliefs])
    reveal_best_outcome!(b1, s1, m1.n_gamble)

    s1, b1
end

function simulate_suggest_after(pol, s, b)
    # first decision
    choice, payoff, cost = simulate(pol, s, b)
    s1, b1 = add_new_option(s, b)
    # second decision
    choice, payoff, cost1 = simulate(pol, s1, b1)
    cost += cost1
    (;choice, payoff, cost)
end


function sample_suggest_new_effect(m, polclass=DCPolicy)
    pol = polclass(m)
    s = experiment_state(m)
    b = Belief(s)

    before = simulate(pol, add_new_option(s, b)...)
    after = simulate_suggest_after(pol, s, b)

    features = (;
        weight_dev = sum(abs.(s.weights .- mean(s.weights)))
    )

    (suggestion = m.n_gamble + 1, features, before, after)
end
