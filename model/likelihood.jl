include("timeout.jl")

using StatsFuns: logsumexp

function choice_likelihood(s::State, b=Belief(s), post_decision=nothing)

    # For efficient cache lookup, we use as a key an unsigned integer where each position in the
    # binary representation is a flag for whether the corresponding cell has been revealed.
    pol = MetaGreedy(s.m)
    flag = UInt64(1)
    @assert length(s.payoffs) < 64  # a 64 bit flag means this is the most cells we can handle

    cache = Dict{UInt64,Vector{Float64}}()
    function recurse(b::Belief, pd::Bool, k::UInt)
        yield()  # allow interruption
        haskey(cache, k) && return cache[k]
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            if !pd  # not already applied
                return cache[k] = recurse(post_decision(b), true, k + 1)
            else
                return cache[k] = choice_probs(b)
            end
        else
            opt_c = findall(isequal(maximum(v)), v)
            probs = mapreduce(+, opt_c) do c
                recurse(observe(b, s, c), pd, k + (flag << c))
            end
            probs /= length(opt_c)
            return cache[k] = probs
        end
    end
    recurse(b, post_decision == nothing, UInt(0))
end

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
    choice_likelihood(s, b)
end

function suggest_late_likelihood(t::Trial; naive::Bool)
    s = State(t)
    post_decision(b) = apply_suggestion!(Belief(s), s, t.nudge_index, naive)
    choice_likelihood(s, Belief(s), post_decision)
end

function control_likelihood(t; kws...)
    choice_likelihood(State(t))
end

function likelihood(t::Trial; max_time=240, kws...)
    like = Dict(
        "control" => control_likelihood,
        "post-supersize" => suggest_late_likelihood,
        "pre-supersize" => suggest_early_likelihood
    )[t.nudge_type]

    try
        timeout(max_time, t) do
            like(t; kws...)
        end
    catch err
        if err isa TimeoutException
            n_opt = size(t.payoffs, 2) 
            ones(n_opt) ./ n_opt
        else
            rethrow()
        end
    end
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
