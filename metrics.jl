include("mouselab.jl")
using Distributed
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

@everywhere function choice_cost_steps(pol::Policy, prob::Problem)
    total_p = 0.
    n_clicks = 0.
    cost = 0.
    choice = zeros(prob.prm.n_gamble)

    function recurse(b, p, n)
        # print("\n>>> ", p, "  ")
        # display(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            n_clicks += p * n
            choice .+= p .* choice_probs(b)
            total_p += p
        else
            opt_c = findall(softmax(1e20 * v) .!= 0)
            p /= length(opt_c)
            for c in opt_c
                cost += p * prob.cost[c]
                recurse(observe(b, prob, c), p, n+1)
            end
        end
    end

    recurse(Belief(prob), 1, 0)
    @assert total_p ≈ 1 "total_p = $total_p"
    return choice, cost, n_clicks
end

expected_reward(prob::Problem) = expected_reward(meta_greedy, prob)

function parallel_expected(f::Function, prob::Problem, pol, n)
    total = @distributed (+) for i in 1:n
        f(rollout(pol, prob))
    end
    total / n
end

function serial_expected(f::Function, prob::Problem, pol, n)
    mean(f(rollout(pol, prob)) for i in 1:n)
end

function expected(f::Function, prob::Problem, pol, n=1000)
    (n > 1000 ? parallel_expected : serial_expected)(f, prob, pol, n)
end

function expected_reward(pol, prob::Problem; n=1000)
    @assert false
    expected(prob, pol, n) do roll
        roll.choice_value - roll.total_cost
    end
end

#
# function choice_probs(pol, prob::Problem)
#     @assert false "WARNING this is incorrect because it ignores probabilities (see above)"
#     counts = zeros(prob.prm.n_gamble)
#     function rec(b)  # FIXME this should take a p argument as well
#         v = voc(pol, b)
#         if all(v .<= 0) # terminate
#             counts .+= choice_probs(b)
#         else
#             for c in findall(softmax(1e20 * v) .!= 0)
#                 rec(observe(b, prob, c))
#             end
#         end
#     end
#     rec(Belief(prob))
#     return counts / sum(counts)
# end
