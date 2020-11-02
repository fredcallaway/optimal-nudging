# include("mouselab.jl")
# using Distributed

function expected_reward(pol::Policy, s::State)
    total_p = 0.
    value = 0.

    function recurse(b, p)
        # print("\n>>> ", p, "  ")
        # display(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            value += p * true_term_reward(s, b)
            total_p += p
        else
            opt_c = findall(softmax(1e20 * v) .!= 0)
            p /= length(opt_c)
            for c in opt_c
                value -= p * s.costs[c]
                recurse(observe(b, s, c), p)
            end
        end
    end

    recurse(Belief(s), 1.)
    @assert total_p ≈ 1 "total_p = $total_p"
    return value
end

@everywhere function choice_cost_steps(pol::Policy, s::State)
    total_p = 0.
    n_clicks = 0.
    cost = 0.
    choice = zeros(s.prm.n_gamble)

    function recurse(b, p, n)
        # print("\n>>> ", p, "  ")
        # display(b)
        v = voc(pol, b)
        if all(v .<= 0) # terminate
            n_clicks += p * n
            choice .+= p .* choice_ss(b)
            total_p += p
        else
            opt_c = findall(softmax(1e20 * v) .!= 0)
            p /= length(opt_c)
            for c in opt_c
                cost += p * s.cost[c]
                recurse(observe(b, s, c), p, n+1)
            end
        end
    end

    recurse(Belief(s), 1, 0)
    @assert total_p ≈ 1 "total_p = $total_p"
    return choice, cost, n_clicks
end

expected_reward(s::State) = expected_reward(meta_greedy, s)

function parallel_expected(f::Function, s::State, pol, n)
    total = @distributed (+) for i in 1:n
        f(rollout(pol, s))
    end
    total / n
end

function serial_expected(f::Function, s::State, pol, n)
    mean(f(rollout(pol, s)) for i in 1:n)
end

function expected(f::Function, s::State, pol, n=1000)
    (n > 1000 ? parallel_expected : serial_expected)(f, s, pol, n)
end

function expected_reward(pol, s::State; n=1000)
    @assert false
    expected(s, pol, n) do roll
        roll.choice_value - roll.total_cost
    end
end

#
# function choice_ss(pol, s::State)
#     @assert false "WARNING this is incorrect because it ignores sabilities (see above)"
#     counts = zeros(s.prm.n_gamble)
#     function rec(b)  # FIXME this should take a p argument as well
#         v = voc(pol, b)
#         if all(v .<= 0) # terminate
#             counts .+= choice_ss(b)
#         else
#             for c in findall(softmax(1e20 * v) .!= 0)
#                 rec(observe(b, s, c))
#             end
#         end
#     end
#     rec(Belief(s))
#     return counts / sum(counts)
# end
