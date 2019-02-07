include("mouselab.jl")
const _meta_greedy = Policy([0,1,0,0,0])

function choice_probs(pol::Policy, prob::Problem)
    @assert false "this is incorrect! See function below for reference"
    counts = zeros(prob.prm.n_gamble)
    function rec(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            counts .+= choice_probs(b)
        else
            for c in findall(softmax(1e20 * v) .!= 0)
                rec(observe(b, prob, c))
            end
        end
    end
    rec(Belief(prob))
    return counts / sum(counts)
end

function expected_reward(pol::Policy, prob::Problem)
    total_p = 0
    value = 0

    function recurse(b, p)
        # print("\n>>> ", p, "  ")
        # display(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            value += p * true_term_reward(prob, b)
            total_p += p
        else
            opt_c = findall(softmax(1e20 * v) .!= 0)
            p /= length(opt_c)
            for c in opt_c
                value -= p * prob.cost[c]
                recurse(observe(b, prob, c), p)
            end
        end
    end

    recurse(Belief(prob), 1)
    @assert total_p ≈ 1 "total_p = $total_p"
    return value
end
expected_reward(prob::Problem) = expected_reward(_meta_greedy, prob)

# %% ====================  ====================
# prm = Params(n_gamble=6, n_outcome=4, cost=0.01)
# prob = Problem(prm)
# pol = Policy([0,1,0,0,0])
# @time expected_reward(pol, prob);
# println(mean(rollout(pol, prob).assistant_expected_reward for i in 1:1000))
# println(expected_reward(pol, prob))
