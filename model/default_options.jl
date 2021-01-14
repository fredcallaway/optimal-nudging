
function estimate_default_beliefs(m; N=1000000)
    default_vals = Float64[]; other_vals = Float64[]
    for i in 1:N
        payoffs = rand(m.reward_dist, (m.n_feature, m.n_option))
        default = argmax(sum(payoffs; dims=1)).I[2]
        for option in 1:m.n_option
            lst = option == default ? default_vals : other_vals
            for feature in 1:m.n_feature
                push!(lst, payoffs[feature, option])
            end
        end
    end
    (default=fit(Normal, default_vals), other=fit(Normal, other_vals))
end

if !@isdefined(DEFAULT_BELIEFS)
    DEFAULT_BELIEFS = Dict{UInt64,NamedTuple{(:default, :other),Tuple{Normal{Float64},Normal{Float64}}}}()
end

function apply_default!(b::Belief, default::Int; use_cache=true)
    D = use_cache ? DEFAULT_BELIEFS[hash(b.m)] : estimate_default_beliefs(b.m)
    for option in 1:b.m.n_option
        d = option == default ? D.default : D.other
        b.matrix[:, option] .= b.s.weights .* d
    end
    b
end

function simulate_default(pol, s, b, default)
    choice, payoff, cost, clicks = simulate(pol, s, b)
    default_clicks = LinearIndices(b.matrix)[:, default]
    (
        n_click_default = isempty(clicks) ? 0 : sum(c in default_clicks for c in clicks),
        decision_cost = cost,
        payoff,
        choose_default = Int(choice == default),
    )
end

function sample_default_effect(m::MetaMDP, polclass=MetaGreedy)
    pol = polclass(m)
    s = experiment_state(m)
    b = Belief(s)
    default = argmax(sum(s.payoffs; dims=1)).I[2]
    nudged_b = apply_default!(Belief(s), default)

    (default,
     weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     with = simulate_default(pol, s, nudged_b, default),
     without = simulate_default(pol, s, b, default))
end
