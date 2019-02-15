using Distributed
using Random
using StatsBase

pwd()
addprocs(40)
@everywhere cd("/usr/people/flc2/juke/cost-modification")
@everywhere include("cost_modification.jl")

# %% ====================  ====================

function tournament(fitness)
    chosen = sample(1:pop_size, 2; replace=false)
    argmax(fitness[chosen])
end

function mutate!(x)
    pos = rand(1:length(x))
    x[pos] = !x[pos]
    x
end
mutate(x) = mutate!(copy(x))
length(workers())

# %% ====================  ====================
prm = Params(n_gamble=6,n_outcome=3,cost=0.05, weight_alpha=1)
prob = Problem(prm)
policy = NoisyMetaGreedy(50)
expected_reward(policy, prob)
expected_reward(meta_greedy, prob)

objective = make_objective(prob, policy)
println("-"^70)
display(prob)
println("  No sale: ", objective(falses(length(prob.matrix))))
println("Best cell: ", objective(best_cell))
println("Greedy: ", objective(greedy_select(prob)))


function computation_log(policy, prob)
    cs = Int[]
    rollout(policy, sale(prob, x), callback=(b, c)->push!(cs, c))
    cs
end

@time x = greedy_select(prob);

x[1] = true
expected(sale(prob, x), policy) do roll
    roll.choice_value - roll.total_cost
end

make_objective(prob, policy, n=10000)(best_cell);
println(join(computation_log(policy, sale(prob, x)), " "))
countmap([computation_log(policy, sale(prob, x))[1] for i in 1:1000])


reshape(x, size(prob.matrix))
reshape(1:length(prob.matrix), size(prob.matrix))

function evolve_sale(prob, policy;
                     p_mutate = 0.2,
                     pop_size = 100,  # try to make this a multiple of length(workers())
                     n_elite = 10,
                     n_iter = 100,
                     verbosity = 0,
                     parallelize=true)

    @everywhere objective = make_objective($prob, $policy)
    rand_individual() = bitrand(length(prob.cost))
    population = [rand_individual() for i in 1:pop_size]
    offspring = copy(population)
    mymap = parallelize ? pmap : map
    for iter in 1:n_iter
        fitness = mymap(objective, population)
        elite = sortperm(fitness; rev=true)[1:n_elite]
        if verbosity > 0 && iter % verbosity == 0
            println("Iteration $iter")
            println("  Best fitness: $(fitness[elite[1]])")
            println("      # unique: $(length(unique(population)))")
        end
        # println("Mean: $(mean(fitness))")
        for i in 1:n_elite
            offspring[i] = population[elite[i]]
        end
        for i in n_elite+1:pop_size
            x = population[tournament(fitness)]
            if rand() < p_mutate
                x = mutate(x)
            end
            offspring[i] = x
        end
        population .= offspring
    end
    fitness = pmap(objective, population)
    best = unique(population[fitness .== maximum(fitness)])
    best
end

@time evolve_sale(prob)

fitness = pmap(objective, population)
best = unique(population[fitness .== maximum(fitness)])
rollout(meta_greedy, sale(prob, best_cell), callback=(b,c)->display(b)).choice
rollout(meta_greedy, sale(prob, best[1]), callback=(b,c)->display(b)).choice

# function show_sale(prob, select)


# %% ====================  ====================

NoisyPolicy()



# population = population[selected]

# %% ====================  ====================

print_result(no_sale, prob)
print_result(best_cell_select, prob)
print_result(greedy_select, prob)
x = greedy_select(prob; init=true)

expected_reward(meta_greedy, prob)

undo! = sale!(prob, x)
println(expected_reward(meta_greedy, prob))
undo!()

undo! = sale!(prob, best_cell_select(prob))
println(expected_reward(meta_greedy, prob))
rollout(meta_greedy, prob).belief
undo!();

# %% ====================  ====================
best, invbestfit, _, _, history = ga(x->float(sum(x)),
          5,
          initPopulation=n->rand(0:1, n),
          iterations=100,
          mutation = inversion,
          mutationRate = 0.2,
          selection = tournament(4),
          populationSize = 50,
          # ε=2,
          interim=true,
          debug=false
          )
println(best, "  ", invbestfit)


# %% ====================  ====================
using StatsBase
x = Float64[1,5,1]
x ./ sum(x)
counts(sus(x, 1000)) ./ 1000

# %% ====================  ====================
prob = Problem(prm)
loss = make_loss(prob)
@time best, invbestfit, generations, tolerance, history = ga(
    x->-loss(x),                    # Function to MINIMISE
    length(prob.matrix),
    initPopulation = N->rand(Bool, N),
    selection = tournament(2),                   # Options: sus
    mutation = flip,                   # Options:
    crossover = singlepoint,                # Options:
    mutationRate = 0.2,
    crossoverRate = 0.5,
    ɛ = 5,                                # Elitism
    iterations = 500,
    populationSize = 50,
    verbose=false,
    interim = true);

# %% ====================  ====================

print_result(best_cell_select, prob)
print_result(best, prob, "evolved")
# %% ====================  ====================

println(argmax(history[:bestFitness] .== 0))
println(fitness(best))
# length(history[:bestFitness])

# %% ====================  ====================

mass    = [1, 5, 3, 7, 2, 10, 5]
utility = [1, 3, 5, 2, 5,  8, 3]

function fitness(n::AbstractVector)
    total_mass = sum(mass .* n)
    return (total_mass <= 20) ? sum(utility .* n) : 0
end

initpop = collect(rand(Bool,length(mass)))

best, invbestfit, generations, tolerance, history = ga(
    x -> 1 / fitness(x),                    # Function to MINIMISE
    length(initpop),                        # Length of chromosome
    initPopulation = initpop,
    selection = roulette,                   # Options: sus
    mutation = flip,                   # Options:
    crossover = singlepoint,                # Options:
    mutationRate = 0.2,
    crossoverRate = 0.5,
    ɛ = 0.1,                                # Elitism
    iterations = 20,
    populationSize = 50,
    interim = true);
best

@test fitness(best) == 21.
@test 1. /invbestfit == 21.

# %% ====================  ====================

using Test
using Random

N = 8
P = 50
generatePositions(N::Int) = collect(1:N)[randperm(N)]

# Vector of N cols filled with numbers from 1:N specifying row position
function nqueens(queens::Vector{Int})
    n = length(queens)
    fitness = 0
    for i=1:(n-1)
        for j=(i+1):n
            k = abs(queens[i] - queens[j])
            if (j-i) == k || k == 0
                fitness += 1
            end
            # println("$(i),$(queens[i]) <=> $(j),$(queens[j]) : $(fitness)")
        end
    end
    return fitness
end
@test nqueens([2,4,1,3]) == 0
@test nqueens([3,1,2]) == 1

# Testing: GA solution with various mutations
for muts in [inversion, insertion, swap2, scramble, shifting]
    result, fitness, cnt = ga(nqueens, N;
        initPopulation = generatePositions,
        populationSize = P,
        selection = sus,
        crossover = pmx,
        mutation = muts)
    println("GA:PMX:$(string(muts))(N=$(N), P=$(P)) => F: $(fitness), C: $(cnt), OBJ: $(result)")
    @test nqueens(result) == 0
end

# Testing: ES
for muts in [inversion, insertion, swap2, scramble, shifting]
    result, fitness, cnt = es(nqueens, N;
        initPopulation = generatePositions,
        mutation = mutationwrapper(muts),
        μ = 15, ρ = 1, λ = P)
    println("(15+$(P))-ES:$(string(muts)) => F: $(fitness), C: $(cnt), OBJ: $(result)")
    @test nqueens(result) == 0
end
