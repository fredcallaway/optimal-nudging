# include("mouselab.jl")
# include("metrics.jl")

KNOWN_WEIGHTS = false
using Random: shuffle
using SplitApplyCombine

function metagreedy_optimal_clicks(b::Belief)
    v = voc1(b)
    maxv = maximum(v)
    maxv < 0 && return [⊥]
    findall(isequal(maxv), v)
end

function expected_reward(s::State, b=Belief(s), optimal_clicks=metagreedy_optimal_clicks)
    # For efficient cache lookup, we use as a key an unsigned integer where each position in the
    # binary representation is a flag for whether the corresponding cell has been revealed.
    pol = MetaGreedy(s.m)
    flag = UInt64(1)
    @assert length(s.payoffs) < 64  # a 64 bit flag means this is the most cells we can handle

    choice_val = choice_values(s)

    cache = Dict{UInt64,Float64}()
    function recurse(b::Belief, k::UInt)
        yield()  # allow interruption
        haskey(cache, k) && return cache[k]
        # v = voc(pol, b)
        opt_c = optimal_clicks(b)

        if opt_c == [⊥]  # terminate
            return cache[k] = first(choice_val * choice_probs(b))
        else
            # opt_c = findall(isequal(maximum(v)), v)
            probs = mapreduce(+, opt_c) do c
                recurse(observe(b, s, c), k + (flag << c)) - s.costs[c] 
            end
            probs /= length(opt_c)
            return cache[k] = probs
        end
    end
    recurse(b, UInt(0))
end

function exp3_state(;n_option=5, n_feature=5, base_cost=2, reduction=2, n_rand_reduce=5)
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

function make_objective(s::State, reduction::Real; known_weights::Bool, return_sem::Bool=false)
    if known_weights
        objective1(select::BitVector) = expected_reward(reduce_cost(s, select, reduction))
    else
        states = map(1:1000) do i
            State(s.m; payoffs=s.payoffs, costs=s.costs)
        end
        function objective2(select::BitVector)
            rs = [expected_reward(reduce_cost(s1, select, reduction)) for s1 in states]
            return_sem ? (mean(rs), sem(rs)) : mean(rs)
        end
    end
end

# %% ==================== Optimizing ====================

"Greedily searches for cells to put on reduce_cost"
# function greedy_select(s::State, reduction::Real; init=false, max_flips=1000)
#     init = BitVector(fill(init, length(s.costs)))
#     greedy_select(make_objective(s, reduction), init, max_flips)
# end
function greedy_select(s::State, reduction::Real, n_reduce::Int, base_cost::Real;
                       known_weights=KNOWN_WEIGHTS, verbose=false)
    n_reduce == 0 && return greedy_select_no_limit(s, reduction, base_cost; known_weights, verbose)
    x = BitVector(fill(false, length(s.payoffs)))
    objective = make_objective(s, reduction; known_weights)
    
    function reduction_value(i)
        @assert !x[i]
        x[i] = 1
        fx = objective(x)
        x[i] = 0
        return fx
    end

    current_value = objective(x)
    possible = filter(i->s.costs[i] == base_cost, 1:length(s.costs))

    for i in 1:n_reduce
        v = reduction_value.(possible)
        maxv = maximum(v)
        # maxv < current_value && return x  # don't make things worse
        current_value = maxv
        i = rand(possible[findall(isequal(maxv), v)])
        setdiff!(possible, i)
        x[i] = 1
        verbose && println("($i) ", x)
    end
    return x
end

function greedy_select_no_limit(s::State, reduction::Real, base_cost::Real; 
                                known_weights=KNOWN_WEIGHTS, verbose=false)
    (x1, val1), (x2, val2) = map([true, false]) do init  # start with all or no costs decreased
        x = BitVector(fill(init, length(s.payoffs)))
        objective = make_objective(s, reduction; known_weights, return_sem=true)
        
        if init
            # these can't be changed
            x[(s.costs .!= base_cost)[:]] .= false
        end

        possible = filter(i->s.costs[i] == base_cost, 1:length(s.costs))
        current_value = objective(x)[1]
        seen = Set{BitVector}()
        
        function flip_value(i)
            x[i] = !x[i]
            fx = objective(x)
            x[i] = !x[i]
            return fx
        end

        function do_flip!(could_flip)
            for i in shuffle(could_flip)
                x[i] = !x[i]
                x ∉ seen && return true
                x[i] = !x[i]  # flip back
            end
            return false
        end

        for iter in 1:50
            push!(seen, x)
            vse = flip_value.(possible)  # how much does objective improve from adding/removing the extra cost at each cell
            v, se = invert(vse)
            maxv, maxi = findmax(v)
            maxv - se[maxi] <= current_value && return x, current_value  # can't improve
            could_flip = possible[findall(isequal(maxv), v)]

            flipped = do_flip!(could_flip)
            !flipped && return x, current_value
            current_value = maxv
            verbose && println("($iter)  $(sum(x))  $(round(current_value; digits=3))  ", x)
        end
        return x, current_value
    end
    # @info "done" val1 val2
    val1 > val2 ? x1 : x2
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
    extremity .+= 1e-8 .* rand(length(extremity))  # break ties randomly
    chosen = partialsortperm(extremity, 1:n_reduce; rev=true)
    manyhot(length(payoffs), chosen)
end

reducible(costs, reduction) = filter(i->costs[i] >= reduction, 1:length(costs))

function random_select(costs, reduction, n_reduce)
    chosen = sample(reducible(costs, reduction), n_reduce; replace=false)
    manyhot(length(costs), chosen)
end

extreme_select(s::State, args...) = extreme_select(s.m, s.payoffs, s.costs, args...)
greedy_select(s::State, args...) = greedy_select(s.m, s.payoffs, s.costs, args...)


function sample_cost_reduction_trial(;base_cost=2, reduction=2, n_reduce=5, n_rand_reduce=5)
    # fix random_select
    s = exp3_state(;base_cost, reduction, n_rand_reduce)
    selections = reduction < 0 ? (
        greedy = greedy_select(s, reduction, n_reduce, base_cost),
        none = BitVector(fill(false, length(s.costs))),
    ) : (
        none = BitVector(fill(false, length(s.costs))),
        random = random_select(s.costs, reduction, n_reduce),
        extreme = extreme_select(s, reduction, n_reduce),
        greedy = greedy_select(s, reduction, n_reduce, base_cost),
    )
    alt_costs = map(selections) do sel
        reduce_cost(s, sel, reduction).costs
    end
    s, alt_costs
end

