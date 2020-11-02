using Optim

println("TODO: Check that this is correct fitting.jl line 3")
State(m::MetaMDP, t::Trial) = s = State(m, t.weights, t.values, m.cost .* t.costs)

function build_voc_table(pol, trials::Vector{Trial})
    V = Vector{Float64}[]
    cs = Int[]
    for t in trials
        s = State(pol.m, t)
        b = Belief(s)

        for c in t.uncovered
            v = [voc(pol, b); 0.]  # add voc(⊥) to end
            push!(V, v)
            push!(cs, c)
            observe!(b, s, c)
        end
        # choice to terminate
        push!(V, [voc(pol, b); 0.])
        push!(cs, length(b.matrix)+1)
    end
    V, cs
end

function logp(V, cs, α)
    mapreduce(+, zip(V, cs)) do (v, c)
        p = softmax(α .* v)
        log(p[c])
    end
end

function softmax_mle(pol, trials)
    V, cs = build_voc_table(pol, trials)
    res = optimize(-20, 20) do log_α
        -logp(V, cs, exp(log_α))
    end
    (α=exp(res.minimizer), logp=-res.minimum)
end



# function logp(V, cs, α, ε)
#     mapreduce(+, zip(V, cs)) do (v, c)
#         p_soft = softmax(α .* v)
#         p = ε * p_rand(prefs) + (1-ε) * p_soft
#         log(p[c])
#     end
# end

# function error_mle(pol, trials)
#     V, cs = build_voc_table(pol, trials)
#     res = optimize(-20, 20) do log_α
#         -logp(V, cs, exp(log_α), exp(ε))
#     end
#     (α=exp(res.minimizer), logp=-res.minimum)
# end

# function fit_error_model(model::Model, data::Vector{Datum}; x0 = [0.002, 0.1], biased=false)
#     biased && return fit_biased_error_model(model, data)
#     lower = [1e-3, 1e-3]; upper = [10., 1.]
#     all_prefs = [preferences(model, d) for d in data]
#     cs = [d.c for d in data]
#     opt = optimize(lower, upper, x0, Fminbox(LBFGS())) do (α, ε)
#         - mapreduce(+, all_prefs, cs) do prefs, c
#             logp(prefs, c, α, ε)
#         end
#     end
#     (α=opt.minimizer[1], ε=opt.minimizer[2], logp=-opt.minimum)
# end