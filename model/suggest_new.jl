function reveal_best_outcome!(b::Belief, s::State, g::Int)
    o = argmax(s.payoffs[:, g])
    cell = LinearIndices(b.matrix)[o, g]
    observe!(b, s, cell)
end

function approximate_truncate(d::Normal, l, u)
    if l == u
        return Normal(l, σ_OBS)
    end
    td = Truncated(d, l, u)
    Normal(mean(td), std(td))
end

function add_new_option(s::State, b::Belief, naive::Bool)
    # TODO: maybe prevent clicking the other ones
    m1 = mutate(s.m, n_gamble=s.m.n_gamble+1)

    # New state
    new_payoffs = round_payoffs!(rand(m1.reward_dist, m1.n_outcome))
    payoffs = [s.payoffs new_payoffs]
    s1 = State(m1, s.weights, payoffs, [s.costs m1.cost .* ones(m1.n_outcome)])

    # New belief
    best_feature_val, best_feature = findmax(new_payoffs)

    new_beliefs = map(eachindex(s.weights), s.weights, new_payoffs) do o, w, payoff
        if o == best_feature
            Normal(w * payoff, σ_OBS)
        else
            if naive
                w * s.m.reward_dist
            else
                # account for the fact that the revealed feature is the best one
                w * approximate_truncate(s.m.reward_dist, 0, best_feature_val)
            end
        end
    end
    b1 = Belief(m1, s1, [b.matrix new_beliefs])

    s1, b1
end

function simulate_suggest_after(pol::Policy, s::State, b::Belief, naive::Bool)
    # first decision
    choice, payoff, cost = simulate(pol, s, b)
    s1, b1 = add_new_option(s, b, naive)
    # second decision
    choice, payoff, cost1 = simulate(pol, s1, b1)
    cost += cost1
    (;choice, payoff, cost)
end

function simulate_suggest(pol, s, b, naive, after)    
    choice, payoff, cost = after ?
      simulate_suggest_after(pol, s, b, naive) :
      simulate(pol, add_new_option(s, b, naive)...)


    (naive=Int(naive), after=Int(after), decision_cost=cost, payoff,
     weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     choose_suggested = Int(choice == s.m.n_gamble+1))
end

function sample_suggest_new_effect(m, polclass=DCPolicy)
    pol = polclass(m)
    s = experiment_state(m)

    [simulate_suggest(pol, s, Belief(s), naive, after) 
       for naive in [false, true] for after in [false, true]]
end
