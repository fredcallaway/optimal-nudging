# include("mouselab.jl")
# include("metrics.jl")

function modify_cost!(s::State, select, f::Function)
    old = s.costs[select]  # copys data (not a view)
    s.costs[select] .= f.(s.costs[select])
    undo!() = s.costs[select] .= old  # this is silly
    return undo!
end

function sale!(s::State, select, budget)
    sale_per_cell = budget / sum(select)
    modify_cost!(s, select, cost -> max(0, cost - sale_per_cell))
end
function sale(s::State, select, budget)
    s = deepcopy(s)
    sale!(s, select, budget)
    s
end

function make_objective(s::State, budget::Float64; make_pol=MetaGreedy, n=nothing)
    pol = make_pol(s.m)
    objective(select::BitVector) = evaluate(pol, sale(s, select, budget)).meta_return
end


function make_objective(m::MetaMDP, payoffs::Matrix, budget::Float64; make_pol=MetaGreedy, n_weight=1000)
    pol = make_pol(m)
    # states = [State(m; payoffs=payoffs) for i in 1:n_weight]
    states = map(1:n_weight) do s
        s = State(m; payoffs=payoffs)
        s.weights .+= .001 .* randn(length(s.weights))
        s
    end
    objective(select::BitVector) = mean(evaluate(pol, sale(s, select, budget)).meta_return for s in states)
end

# function make_loss(s::State, budget::Float64=0.1)
#     objective = make_objective(prob, budget)
#     return (select::BitVector) -> begin
#         return -objective(select)
#     end
# end

function advertise!(s::State; factor::Float64=2.)
    for col in 1:size(prob.matrix, 2)
        row = argmax(prob.matrix[:, col])
        s.costs[row, col] /= 2
    end
end

# %% ==================== Heuristic strategies ====================

"Puts the highest-value cell of the best choice on sale"
function best_cell_select(s::State)
    select = falses(length(s.costs))
    best_col = argmax((prob.weights' * prob.matrix)[:])
    best_row = argmax(prob.weights .* prob.matrix[:, best_col])
    best_cell = reshape(1:length(prob.matrix), size(prob.matrix))[best_row, best_col]
    select[best_cell] = true
    select
end

"Greedily searches for cells to put on sale"
function greedy_select(s::State, budget::Float64; init=false, max_flips=1000)
    init = BitVector(fill(init, length(s.costs)))
    greedy_select(make_objective(s, budget), init, max_flips)
end
function greedy_select(m::MetaMDP, payoffs::Matrix, budget::Float64; init=false, max_flips=1000)
    init = BitVector(fill(init, length(payoffs)))
    greedy_select(make_objective(m, payoffs, budget), init, max_flips)
end

function greedy_select(objective::Function, init, max_flips)
    x = deepcopy(init)
    best = objective(x)
    try_flip(i) = begin
        x[i] = !x[i]
        l = objective(x)
        x[i] = !x[i]
        return l
    end
    for iter in 1:max_flips
        println("($iter) ", x)
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
backwards_greedy(s::State) = greedy_select(s; init=true)
