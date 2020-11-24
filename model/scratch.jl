include("nudging_base.jl")
m = MetaMDP(5, 3, Normal(5, 1.5), Dirichlet(ones(3)), 0.1)
# %% --------
s = round_payoffs!(State(m))
b = Belief(s)
pol = DCPolicy(m)
choice, payoff, cost = simulate(pol, s, b)
# %% --------



# %% ==================== Stats ====================
using RCall

for k in [:many_options, :many_features, :high_cost, :nudge]
    dk = data[k]
    data[k] = Int.(dk)
    # data[k] = (float.(dk) .- mean(dk)) ./ std(dk)
end

@rput data


# %% --------
R"""
m = glm(choose_default ~ nudge * (many_options + many_features + high_cost), data=data)
summary(m)
"""
# %% --------
# (Intercept)         0.739212   0.001498  493.55   <2e-16 ***
# nudge:many_options  0.070413   0.001498   47.01   <2e-16 ***
# nudge:many_features 0.042388   0.001498   28.30   <2e-16 ***
# nudge:high_cost     0.081089   0.001498   54.14   <2e-16 ***
# %% --------
R"""
options(width = 160)
m = glm(choose_default ~ nudge : (many_options * many_features * high_cost), data=data)
summary(m)
"""
# %% --------
R"""
m = glm(choose_default ~ nudge * many_options * many_features * high_cost, data=data)
summary(m)
"""

# %% --------

X = summarize(default_effects) do x
    [x.with.choice[x.default],  x.without.choice[x.default]]
end

map(M, X) do m, (with, without)
    (;m.n_gamble, m.n_outcome, m.cost, with, without)
end  |> (x->x[:]) |> CSV.write("results/defaults.csv")
