include("mouselab.jl")


pol = Policy([0, 1, 0, 0, 0])

prm = Params(n_gamble=8, n_outcome=3, cost=.05)
problem = Problem(prm)
problem.matrix

sale = zeros(length(problem.matrix))
sale[2] = 0.05
sale = reshape(sale, prm.n_outcome, prm.n_gamble)
p2 = deepcopy(problem)
p2.cost .-= sale
p2.cost

function computation_order(p2)
    cs = []
    rollout(pol, p2, callback=(b, c) -> push!(cs, c))
    cs
end

@everywhere choice(b::Belief) = argmax(mean.(gamble_values(b)))

function sale_value(sale)
    new_problem = nothing
    roll = rollout(pol, new_problem)
    true_values = new_problem.weights' * new_problem.matrix
    chosen_value = choice_values[choice(roll.belief)]
end

function action_distribution(p::Problem; n_roll=100)
    all_counts = pmap(1:4) do i
        choice_counts = zeros(prm.n_gamble)
        for i in 1:Int(n_roll/4)
            term_choice = rollout(pol, p).choice
            choice_counts[term_choice] += 1
        end
        choice_counts
    end
    sum(all_counts) ./ n_roll
end

@time pmap(1:12) do i
    sleep(1)
end;

@time action_distribution(p2; n_roll=1000);

@everywhere include("mouselab.jl")
using Distributed
addprocs(4)
fetch(@spawnat 2 prm)
