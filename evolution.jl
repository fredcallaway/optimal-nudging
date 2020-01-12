using Distributed
using Random
using StatsBase

# @everywhere include("cost_modification.jl")
# Depends on cost_modification

function tournament(fitness)
    chosen = sample(1:length(fitness), 2; replace=false)
    argmax(fitness[chosen])
end

function mutate!(x)
    pos = rand(1:length(x))
    x[pos] = !x[pos]
    x
end
mutate(x) = mutate!(copy(x))

function evolve_sale(prob, budget, policy;
                     p_mutate = 0.2,
                     pop_size = 100,  # try to make this a multiple of length(workers())
                     n_elite = 10,
                     n_iter = 100,
                     verbosity = 0,
                     parallelize=true)

    @everywhere objective = make_objective($prob, $budget, $policy)
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
