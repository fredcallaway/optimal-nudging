function evaluate(pol::Policy, s::State, b=Belief(s))
    total_p = 0.
    n_clicks = 0.
    cost = 0.
    choice = zeros(s.m.n_gamble)

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
                cost += p * s.costs[c]
                recurse(observe(b, s, c), p, n+1)
            end
        end
    end

    recurse(b, 1, 0)
    @assert total_p ≈ 1 "total_p = $total_p"
    choice_val = only(choice_values(s) * choice)
    meta_return = choice_val - cost
    (;choice, cost, n_clicks, choice_val, meta_return)
end
