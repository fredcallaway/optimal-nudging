include("mouselab.jl")
include("metrics.jl")

const meta_greedy = Policy([0,1,0,0,0])

function modify_cost!(prob::Problem, select::Vector{Bool}, f::Function)
    old = prob.cost[select]  # copys data (not a view)
    prob.cost[select] .= f.(prob.cost[select])
    undo!() = prob.cost[select] .= old  # this is silly
    return undo!
end

function make_loss(prob::Problem; budget::Float64=0.1)
    return (select::Vector{Bool}) -> begin
        sale_per_cell = budget / sum(select)
        undo! = modify_cost!(prob, select, cost -> max(0, cost - sale_per_cell))
        er = expected_reward(meta_greedy, prob)
        undo!()
        return -er
    end
end


# %% ==================== Heuristic strategies ====================

"Puts the highest-value cell of the best choice on sale"
function best_cell_select(prob::Problem)
    select = zeros(Bool, length(prob.cost))
    best_col = argmax((prob.weights' * prob.matrix)[:])
    best_row = argmax(prob.matrix[:, best_col])
    best_cell = reshape(1:length(prob.matrix), size(prob.matrix))[best_row, best_col]
    select[best_cell] = true
    select
end

"Greedily searches for cells to put on sale"
function greedy_select(prob::Problem; init=false)
    loss = make_loss(prob)
    x = fill(init, length(prob.cost))
    current_loss = loss(x)
    try_flip(i) = begin
        x[i] = !x[i]
        l = loss(x)
        x[i] = !x[i]
        return l
    end
    while true
        new_loss, i = findmin(try_flip.(1:length(x)))
        if new_loss < current_loss
            current_loss = new_loss
            x[i] = !x[i]
        else
            break
        end
    end
    return x
end
backwards_greedy(prob::Problem) = greedy_select(prob; init=true)
