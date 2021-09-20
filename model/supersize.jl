function reveal_best_feature!(b::Belief, s::State, option::Int)
    feature = argmax(s.payoffs[:, option])
    cell = LinearIndices(b.matrix)[feature, option]
    observe!(b, s, cell)
end

function approximate_truncate(d::Normal, l, u)
    if l == u
        return Normal(l, σ_OBS)
    end
    td = Truncated(d, l, u)
    Normal(mean(td), std(td))
end

function supersize_beliefs(weights::Vector, reward_dist::Normal, payoffs::Vector, naive::Bool)
    best_feature_val, best_feature = findmax(payoffs)
    map(eachindex(weights), weights, payoffs) do feature, w, payoff
        if feature == best_feature
            Normal(w * payoff, σ_OBS)
        else
            if naive
                w * reward_dist
            else
                # account for the fact that the revealed feature is the best one
                w * approximate_truncate(reward_dist, 0, best_feature_val)
            end
        end
    end
end

function add_new_option(s::State, b::Belief, naive::Bool)
    # TODO: maybe prevent clicking the other ones
    m1 = mutate(s.m, n_option=s.m.n_option+1)

    # New state
    new_payoffs = round_payoffs!(rand(m1.reward_dist, m1.n_feature))
    payoffs = [s.payoffs new_payoffs]
    s1 = State(m1, s.weights, payoffs, [s.costs m1.cost .* ones(m1.n_feature)])

    # New belief
    new_beliefs = supersize_beliefs(s.weights, s.m.reward_dist, new_payoffs, naive)
    b1 = Belief(m1, s1, [b.matrix new_beliefs])

    s1, b1
end

function simulate_supersize_after(pol::Policy, s::State, b::Belief, naive::Bool)
    # first decision
    choice, payoff, cost, clicks = simulate(pol, s, b)
    s1, b1 = add_new_option(s, b, naive)
    # second decision
    choice, payoff, cost1, clicks1 = simulate(pol, s1, b1)
    cost += cost1
    append!(clicks, clicks1)
    (;choice, payoff, cost, clicks)
end

function simulate_supersize(pol, s, b, naive, after)    
    choice, payoff, cost, clicks = after ?
      simulate_supersize_after(pol, s, b, naive) :
      simulate(pol, add_new_option(s, b, naive)...)

    (naive=Int(naive), after=Int(after), decision_cost=cost, payoff, 
     weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     n_click = length(clicks),
     choose_suggested = Int(choice == s.m.n_option+1))
end

function sample_supersize_effect(m, polclass=MetaGreedy)
    pol = polclass(m)
    s = experiment_state(m)

    [simulate_supersize(pol, s, Belief(s), naive, after) 
       for naive in [true] 
        for after in [false, true]]
end
