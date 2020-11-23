using Distributed
using SplitApplyCombine

addprocs(24)
# @everywhere cd("/usr/people/flc2/juke/cost-modification")
@everywhere include("cost_modification.jl")
@everywhere include("evolution.jl")
println("Ready!")
# %% ====================  ====================

function computation_log(policy, prob)
    cs = Int[]
    rollout(policy, prob, callback=(b, c)->push!(cs, c))
    cs
end

function describe_rollout(policy, prob)
    print("Clicks: ")
    roll = rollout(policy, prob, callback=(b, c)->print(c, " "))
    println()
    println("Choice: ", roll.choice_probs)
    # println("Value: ", round(roll.choice_value - roll.total_cost; digits=3))
end

@everywhere function evaluate(prob)
    pr, cost, ns = choice_cost_steps(policy, prob)
    cv = prob.weights' * prob.matrix
    loss = maximum(cv) - cv * pr
    round.((loss, cost, ns); digits=3)
end


# %% ====================  ====================
display("")
cost = 0.01
budget = cost*3
prm = Params(n_gamble=6, n_outcome=3, cost=cost,
             reward_dist=Normal(0.75, 0.25), weight_alpha=1)
prob = Problem(prm)
policy = meta_greedy
objective = make_objective(prob, budget, policy)
@time select = greedy_select(prob, budget, policy)
# show_cost(sale(prob, select, budget))
describe_rollout(policy, sale(prob, select, budget))
# evaluate(prob)
# evaluate(sale(prob, select, budget))
# @time sales = evolve_sale(prob, policy; verbosity=0, n_iter=100)


# %% ====================  ====================

results = map(.01:.01:.05) do cost
    prm = Params(n_gamble=6, n_outcome=3, cost=cost,
                 reward_dist=Normal(0.75, 0.25), weight_alpha=1)
    budget = 3 * cost
    @time pmap(1:100) do i
        prob = Problem(prm)
        objective = make_objective(prob, budget, policy)
        v1 = evaluate(prob)
        select = greedy_select(prob, budget, policy)
        # select = best_cell_select(prob)
        # select = falses(length(prob.matrix))
        v2 = evaluate(sale(prob, select, budget))
        ((v2 .- v1)..., sum(select))
    end |> invert .|> mean
end


# %% ====================  ====================
prm = Params(n_gamble=6, n_outcome=3, cost=cost,
             reward_dist=Normal(0.75, 0.25), weight_alpha=1)
prob = Problem(prm)

objective = make_objective(prob, 0.03, policy)
x = greedy_select(prob, 0.03, policy)

describe_rollout(policy, prob)
choice_cost_steps(policy, prob)
evaluate(prob)

prob1 = sale(prob, x, budget)
evaluate(prob1)


# %% ====================  ====================
evaluate(prob1)
show_cost(prob1)
describe_rollout(policy, prob1)

x1 = copy(x)
x1[3] = true
prob2 = sale(prob, x1)

# %% ====================  ====================
results



# %% ====================  ====================
objective = make_objective(prob, policy)
greedy = greedy_select(prob, policy)
best_cell = best_cell_select(prob)
println("-"^70)
display(prob)
println("  No sale: ", objective(falses(length(prob.matrix))))
println("Best cell: ", objective(best_cell))
println("   Greedy: ", objective(greedy))
println("  Evolved: ", objective(sales[1]))

prob2 = sale(prob, sales[1])
describe_rollout(policy, prob)
roll = rollout(policy, prob, callback=(b, c)->print(c, " "))


# %% ====================  ====================
using Serialize
using JSON




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
