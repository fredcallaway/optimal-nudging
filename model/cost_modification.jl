# include("mouselab.jl")
# include("metrics.jl")

function exp3_state(;n_option=5, n_feature=5, base_cost=3, reduction=2, n_rand_reduce=5)
    m = MetaMDP(n_option, n_feature, REWARD_DIST, WEIGHTS(n_feature), base_cost)
    s = State(m)
    round_payoffs!(s)
    s.costs[random_select(s.costs, reduction, n_rand_reduce)] .-= reduction
    s
end

function reduce_cost!(s::State, select, reduction)
    s.costs[select] .-= reduction
    s
end

function reduce_cost(s::State, select, reduction)
    reduce_cost!(deepcopy(s), select, reduction)
end

# function make_objective(s::State, reduction::Real; make_pol=MetaGreedy, n=nothing)
#     pol = make_pol(s.m)
#     objective(select::BitVector) = evaluate(pol, reduce_cost(s, select, reduction)).meta_return
# end

function make_objective(m::MetaMDP, payoffs::Matrix, costs::Matrix, reduction::Real; make_pol=MetaGreedy, n_weight=1000)
    pol = make_pol(m)
    # states = [State(m; payoffs=payoffs) for i in 1:n_weight]
    states = map(1:n_weight) do s
        s = State(m; payoffs=payoffs, costs=costs)
        s.weights .+= .001 .* randn(length(s.weights))
        s
    end
    objective(select::BitVector) = mean(evaluate(pol, reduce_cost(s, select, reduction)).meta_return for s in states)
end

# %% ==================== Optimizing ====================

"Greedily searches for cells to put on reduce_cost"
# function greedy_select(s::State, reduction::Real; init=false, max_flips=1000)
#     init = BitVector(fill(init, length(s.costs)))
#     greedy_select(make_objective(s, reduction), init, max_flips)
# end
function greedy_select(m::MetaMDP, payoffs::Matrix, costs::Matrix, reduction::Real, n_reduce::Int; verbose=false)
    init = BitVector(fill(false, length(payoffs)))
    objective = make_objective(m, payoffs, costs, reduction)
    x = deepcopy(init)
    # best = objective(x)
    
    function reduction_value(i)
        @assert !x[i]
        x[i] = 1
        fx = objective(x)
        x[i] = 0
        return fx
    end

    possible = reducible(costs, reduction)

    for i in 1:n_reduce
        i = sample(possible, Weights(softmax(1e5 .* reduction_value.(possible))))
        deleteat!(possible, findfirst(isequal(i), possible))
        x[i] = 1
        verbose && println("($i) ", x)
    end
    return x
end
backwards_greedy(s::State) = greedy_select(s; init=true)

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

function manyhot(N, hot)
    x = BitVector(fill(false, N))
    x[hot] .= true
    x
end

function extreme_select(m, payoffs, costs, reduction, n_reduce)
    extremity = abs.(payoffs[:] .- m.reward_dist.μ)
    possible = reducible(costs, reduction)
    for i in eachindex(extremity)
        if i ∉ possible
            extremity[i] = 0
        end
    end
    chosen = partialsortperm(extremity, 1:n_reduce; rev=true)
    manyhot(length(payoffs), chosen)
end

reducible(costs, reduction) = filter(i->costs[i] >= reduction, 1:length(costs))

function random_select(costs, reduction, n_reduce)
    chosen = sample(reducible(costs, reduction), n_reduce; replace=false)
    manyhot(length(costs), chosen)
end

function get_reductions(s; reduction=2, n_reduce=5)
    selections = (
        none = BitVector(fill(false, length(s.costs))),
        random = random_select(s.costs, reduction, n_reduce),
        extreme = extreme_select(s.m, s.payoffs, s.costs, reduction, n_reduce),
        # greedy = greedy_select(s.m, s.payoffs, s.costs, reduction, n_reduce),
    )
    map(selections) do sel
        reduce_cost(s, sel, reduction)
    end
end
