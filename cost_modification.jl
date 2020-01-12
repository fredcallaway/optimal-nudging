include("mouselab.jl")
include("metrics.jl")

function modify_cost!(prob::Problem, select, f::Function)
    old = prob.cost[select]  # copys data (not a view)
    prob.cost[select] .= f.(prob.cost[select])
    undo!() = prob.cost[select] .= old  # this is silly
    return undo!
end

function sale!(prob, select, budget)
    sale_per_cell = budget / sum(select)
    modify_cost!(prob, select, cost -> max(0, cost - sale_per_cell))
end
function sale(prob, select, budget)
    prob = deepcopy(prob)
    sale!(prob, select, budget)
    prob
end

function make_objective(prob::Problem, budget::Float64, pol=meta_greedy; n=nothing)
    return (select::BitVector) -> begin
        sale_per_cell = budget / sum(select)
        undo! = sale!(prob, select, budget)
        er = n != nothing ? expected_reward(pol, prob, n=n) : expected_reward(pol, prob)
        undo!()
        return er
    end
end

# function make_loss(prob::Problem, budget::Float64=0.1)
#     objective = make_objective(prob, budget)
#     return (select::BitVector) -> begin
#         return -objective(select)
#     end
# end

function advertise!(prob::Problem; factor::Float64=2.)
    for col in 1:size(prob.matrix, 2)
        row = argmax(prob.matrix[:, col])
        prob.cost[row, col] /= 2
    end
end

# %% ==================== Heuristic strategies ====================

"Puts the highest-value cell of the best choice on sale"
function best_cell_select(prob::Problem)
    select = falses(length(prob.cost))
    best_col = argmax((prob.weights' * prob.matrix)[:])
    best_row = argmax(prob.weights .* prob.matrix[:, best_col])
    best_cell = reshape(1:length(prob.matrix), size(prob.matrix))[best_row, best_col]
    select[best_cell] = true
    select
end

"Greedily searches for cells to put on sale"
function greedy_select(prob::Problem, budget::Float64, pol; init=false, max_sale=1000)
    # use exact expected_reward for non-noisy policies
    n = pol isa Policy ? nothing : 10000
    objective = make_objective(prob, budget, pol; n=n)
    x = BitVector(fill(init, length(prob.cost)))
    best = objective(x)
    try_flip(i) = begin
        x[i] = !x[i]
        l = objective(x)
        x[i] = !x[i]
        return l
    end
    for _ in 1:max_sale
        obj, i = findmax(try_flip.(1:length(x)))
        if obj > best
            best = obj
            x[i] = !x[i]
        else
            break
        end
    end
    return x
end
backwards_greedy(prob::Problem) = greedy_select(prob; init=true)
