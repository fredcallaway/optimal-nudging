include("utils.jl")
include("meta_mdp.jl")
include("directed_cognition.jl")
include("meta_greedy.jl")
include("nudging_base.jl")

function estimate_default_beliefs(m; N=1000000)
    default_vals = Float64[]; other_vals = Float64[]
    for i in 1:N
        payoffs = rand(m.reward_dist, (m.n_outcome, m.n_gamble))
        default = argmax(sum(payoffs; dims=1)).I[2]
        for g in 1:m.n_gamble
            lst = g == default ? default_vals : other_vals
            for o in 1:m.n_outcome
                push!(lst, payoffs[o, g])
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
    for g in 1:b.m.n_gamble
        d = g == default ? D.default : D.other
        b.matrix[:, g] .= b.s.weights .* d
    end
    b
end

function sample_default_effect(m::MetaMDP, polclass=DCPolicy)
    pol = polclass(m)
    s = experiment_state(m)
    b = Belief(s)
    default = argmax(sum(s.payoffs; dims=1)).I[2]
    nudged_b = apply_default!(Belief(s), default)
    (default, weight_var = var(s.weights), weight_dev = sum(abs.(s.weights .- mean(s.weights))),
     with = simulate(pol, s, nudged_b), without = simulate(pol, s, b))
end
