include("cost_modification.jl")
using DataStructures

prm = Params(n_gamble=6,n_outcome=3,cost=0.1, weight_alpha=1)
prob = Problem(prm)
original_max = argmax(prob.weights'*prob.matrix)[2]

# Make the matrix bad - this means that the best
# feature of the best option will have a value of amount_below
function make_worse(p::Problem;amount_below=0.1)
    best_choice = argmax(p.weights'*p.matrix)[2]
    best_choice_max_value = maximum(p.matrix[:,best_choice])
    print(best_choice)

    # make all of the best item's values leq amount_below
    new_matrix = p.matrix .- (best_choice_max_value+amount_below)
    new_prob = Problem(p.prm,new_matrix,p.weights,p.cost)
    return new_prob
end


# Make the expected reward of the second best option
# mostly dependent on one value, whereas the EV of the best
# option is evenly distributed among the outcomes.
# May make it so that second best choice is cheaper to compute than
# the best option
function complicate_problem(p::Problem;add_amount=1.75)
    ev_vector = p.weights'*p.matrix
    ev_sorted = sort(ev_vector,dims=2,rev=true)
    best_ev = ev_sorted[1]
    second_ev = ev_sorted[2]
    modifiable_matrix = deepcopy(p.matrix)

    # second-best option
    second_column = findall(ev_vector -> (ev_vector == second_ev),ev_vector)[1][2]
    second_values = modifiable_matrix[:,second_column]
    second_max = argmax(second_values)
    modifiable_matrix[second_max,second_column]+=add_amount
    modified_ev = add_amount*p.weights[second_max]

    # best option
    first_column = argmax(ev_vector)[2]
    modifiable_matrix[:,first_column].+= modified_ev
    new_problem = Problem(p.prm,modifiable_matrix,p.weights,p.cost)
    return new_problem
end


# If complicate then make_worse,
# may make it so assistant helps user choose 2nd best option
prob=complicate_problem(prob)
prob = make_worse(prob,amount_below=0.1)


loss = make_loss(prob)
no_sale(prob) = zeros(Bool, length(prob.cost))

using Printf
print_result(f::Function) = begin
    x = f(prob)
    @printf("%18s: %s   %0.5f\n", f, Int.(x), loss(x))
end
print_result(no_sale)
print_result(best_cell_select)
print_result(greedy_select)
print_result(backwards_greedy)
