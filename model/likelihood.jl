using StatsFuns: logsumexp

function State(t::Trial)
    nf, no = size(t.payoffs)
    m = MetaMDP(no, nf, REWARD_DIST, WEIGHTS(nf), NaN)
    State(m, copy(t.weights), copy(t.payoffs), copy(t.costs))
end

function apply_suggestion!(b, s, nudge_index, naive)
    new_beliefs = suggestion_beliefs(s.weights, s.m.reward_dist, s.payoffs[:, nudge_index], naive)
    b.matrix[:, nudge_index] .= new_beliefs
    b
end

function suggest_early_likelihood(t::Trial; naive::Bool)
    s = State(t)
    b = Belief(s)
    apply_suggestion!(b, s, t.nudge_index, naive)
    evaluate(MetaGreedy(s.m), s, b).choice
end

function suggest_late_likelihood(t::Trial; naive::Bool)
    s = State(t)
    b = Belief(s)
    post_decision(b) = apply_suggestion!(deepcopy(b), s, t.nudge_index, naive)
    evaluate(MetaGreedy(s.m), s, b, post_decision).choice
end

function control_likelihood(t; kws...)
    evaluate(MetaGreedy(s.m), State(t)).choice
end

function likelihood(t::Trial; kws...)
    like = Dict(
        "control" => control_likelihood,
        "post-supersize" => suggest_late_likelihood,
        "pre-supersize" => suggest_early_likelihood
    )[t.nudge_type]
    like(t; kws...)
end

lapse(p, ε) = (1 - ε) .* p .+ ε .* ones(length(p)) ./ length(p)

function logp(likes, choices, ε)
    mapreduce(+, likes, choices) do like, choice
        log(lapse(like, ε)[choice])
    end
end

function marginal_logp(likes, choices)
    εs = 0:.001:1
    lp = [logp(likes, choices, ε) for ε in εs]
    logsumexp(lp) - log(length(εs))
end


function individual_marginal_logp(likes, trials)
    G = group(zip(trials, likes)) do (t, like)                                                                                                    
       t.participant_id                                                                                                          
    end 

    map(G) do tls
        likes = second.(tls)
        choices = getfield.(first.(tls), :choice)
        marginal_logp(likes, choices)::Float64
    end
end

# function map_ε(likes, choices)
#     res = optimize(0, 1) do ε
#         -logp(likes, choices, ε)
#     end
#     res.minimizer
# end
