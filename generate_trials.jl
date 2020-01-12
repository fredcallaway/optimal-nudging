using Distributed
addprocs(4)
@everywhere cd("/usr/people/flc2/juke/cost-modification")
@everywhere include("cost_modification.jl")

using JSON
@everywhere prm = Params(n_gamble=6, n_outcome=3, cost=0.02,
                         reward_dist=Normal(0.75, 0.37), weight_alpha=5)

@everywhere function generate_problem()
    prob = nothing
    while true
        prob = Problem(prm)
        if any(prob.matrix .< 0)
            continue  # inefficient, but it's fast so who cares?
        end
        prob.matrix .= round.(prob.matrix; digits=2)
        prob.weights .= round.(prob.weights; digits=2)
        if sum(prob.weights) != 1
            continue
        end
        prob.cost .+= rand(-prm.cost:.01:prm.cost, size(prob.cost))
        break
    end
    return prob
end

policy = meta_greedy
max_sale = 3
budget = prm.cost * max_sale

# %% ====================  ====================

@everywhere function rand_select(prob, n_sale)
    x = falses(length(prob.cost))
    chosen = sample(1:length(x), n_sale; replace=false)
    x[chosen] .= true
    x
end

@everywhere to_cents(x) = Int(round(x * 100; digits=5))

function write_trials(n; scale=1)
    trials = pmap(1:n) do i
        prob = generate_problem()
        select = greedy_select(prob, budget, policy, max_sale=max_sale)
        # select = best_cell_select(prob)
        (
            problem_id=i,
            payoffs=to_cents.(prob.matrix),
            original_costs=to_cents.(prob.cost),
            sale_costs=to_cents.(sale(prob, select, budget).cost),
            random_costs=to_cents.(sale(prob, rand_select(prob, max_sale), budget).cost),
            weights=to_cents.(prob.weights)
        )
    end
    open("trials.json", "w+") do f
        write(f, json(trials))
    end
end

@time results = write_trials(21; scale=100)
reults.payoffs
results[1]

# %% ====================  ====================



# %% ====================  ====================
r1, r2 = map(results) do r
    prob.matrix .= r.payoffs
    prob.cost .= r.original_costs
    prob.weights .= r.weights
    r1 = expected_reward(prob)
    prob.cost .= r.sale_costs
    r2 = expected_reward(prob)
    (r1, r2)
end |> invert

using HypothesisTests
using RCall
R"""

""

# %% ====================  ====================
display("---")
prm = Params(n_gamble=6, n_outcome=3, cost=0.08,
             reward_dist=Normal(3.00, 1.50), weight_alpha=5)

budget = prm.cost * 3
prob = Problem(prm)
prob.cost .+= rand(-prm.cost:.01:prm.cost, size(prob.cost))
sort!(prob.weights; rev=true)

show_cost(prob)
describe_rollout(policy, prob)
# println(join(collect(0:3:15) .+ argmax(prob.weights), " "))
# println(join(sortperm(prob.cost[:]), " "))

x = greedy_select(prob, budget, policy)
describe_rollout(policy, sale(prob, x, budget))
# show_cost(sale(prob, best_cell_select(prob), budget))

# %% ====================  ====================
# include("evolution.jl")
prob = Problem(prm)
prob.cost .+= rand(-prm.cost:.01:prm.cost, size(prob.cost))

objective = make_objective(prob, budget, policy)
evolved = evolve_sale(prob, budget, policy; pop_size=400, verbosity=10, p_mutate=0.5)
greedy = greedy_select(prob, budget, policy)

println(objective(evolved[1]))
println(objective(greedy))
# %% ====================  ====================
function JSON.lower(prob::Problem)
    (payoffs=round.(prob.matrix; digits=2),
     costs=prob.cost,
     weights=round.(prob.weights; digits=2))
end

problems = [Problem(prm), Problem(prm)]
# isdir("trials") || mkdir("trials")
open("trials.json", "w+") do f
    write(f, json(problems))
end
