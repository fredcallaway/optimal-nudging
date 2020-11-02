using Distributed
nprocs() == 1 && addprocs(4)
# @everywhere cd("/usr/people/flc2/juke/cost-modification")
# @everywhere include("cost_modification.jl")

using JSON

struct WeightDist end
# rand(d::WeightDist) =

@everywhere m = MetaMDP(6, 3, Normal(5, 1.5), Dirichlet(ones(3)), 0.1)
# TODO 1.5 or 1.75
# %% ====================  ====================
@everywhere function generate_problem()
    s = nothing
    while true
        s = State(m)
        if any(s.matrix .< 0)
            continue  # inefficient, but it's fast so who cares?
        end
        # TODO maybe use trials from pilot.

        # s.weights .= round.(s.weights; digits=1) .* 10
        # s.matrix ./= s.weights
        # s.matrix .= round.(s.matrix)
        # s.matrix .*= s.weights
        if sum(s.weights) != 10
            continue
        end
        s.costs .+= rand(-m.cost:.01:m.cost, size(s.costs))
        break
    end
    return s
end

generate_problem()
# %% ====================  ====================
max_sale = 3
budget = m.cost * max_sale

@everywhere function rand_select(s, n_sale)
    x = falses(length(s.costs))
    chosen = sample(1:length(x), n_sale; replace=false)
    x[chosen] .= true
    x
end

@everywhere to_cents(x) = Int(round(x * 100; digits=5))

function write_trials(n; scale=1)
    trials = map(1:n) do i
        s = generate_problem()
        select = greedy_select(s, budget, max_sale=max_sale)
        # select = best_cell_select(s)
        (
            problem_id=i,
            payoffs=to_cents.(s.matrix ./ s.weights),
            original_costs=to_cents.(s.costs),
            sale_costs=to_cents.(sale(s, select, budget).costs),
            random_costs=to_cents.(sale(s, rand_select(s, max_sale), budget).costs),
            weights=to_cents.(s.weights)
        )
    end
    open("trials.json", "w+") do f
        write(f, json(trials))
    end
    trials
end

@time results = write_trials(21; scale=100)
# reults.payoffs
# results[1]

# %% ====================  ====================



# %% ====================  ====================
r1, r2 = map(results) do r
    s.matrix .= r.payoffs
    s.costs .= r.original_costs
    s.weights .= r.weights
    r1 = expected_reward(s)
    s.costs .= r.sale_costs
    r2 = expected_reward(s)
    (r1, r2)
end |> invert

# using HypothesisTests
# using RCall
# R"""

# ""

# # %% ====================  ====================
# display("---")
# m = Params(n_gamble=6, n_outcome=3, cost=0.08,
#              reward_dist=Normal(3.00, 1.50), weight_alpha=5)

# budget = m.cost * 3
# s = State(m)
# s.costs .+= rand(-m.cost:.01:m.cost, size(s.costs))
# sort!(s.weights; rev=true)

# show_cost(s)
# describe_rollout(policy, s)
# # println(join(collect(0:3:15) .+ argmax(s.weights), " "))
# # println(join(sortperm(s.costs[:]), " "))

# x = greedy_select(s, budget, policy)
# describe_rollout(policy, sale(s, x, budget))
# # show_cost(sale(s, best_cell_select(s), budget))

# # %% ====================  ====================
# # include("evolution.jl")
# s = State(m)
# s.costs .+= rand(-m.cost:.01:m.cost, size(s.costs))

# objective = make_objective(s, budget, policy)
# evolved = evolve_sale(s, budget, policy; pop_size=400, verbosity=10, p_mutate=0.5)
# greedy = greedy_select(s, budget, policy)

# println(objective(evolved[1]))
# println(objective(greedy))
# # %% ====================  ====================
# function JSON.lower(s::State)
#     (payoffs=round.(s.matrix; digits=2),
#      costs=s.costs,
#      weights=round.(s.weights; digits=2))
# end

# problems = [State(m), State(m)]
# # isdir("trials") || mkdir("trials")
# open("trials.json", "w+") do f
#     write(f, json(problems))
# end
