include("cost_modification.jl")

prm = Params(n_gamble=6,n_outcome=3,cost=0.1, weight_alpha=1e10)
prob = Problem(prm)
loss = make_loss(prob)
no_sale(prob) = zeros(Bool, length(prob.cost))

using Printf
print_result(f::Function) = begin
    x = f(prob)
    @printf("%18s: %s   %0.3f\n", f, Int.(x), loss(x))
end
print_result(no_sale)
print_result(best_cell_select)
print_result(greedy_select)
print_result(backwards_greedy)
