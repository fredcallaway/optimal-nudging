include("mouselab.jl")
using Combinatorics

# no computation log bc this is from many rollouts, no single computation
# log. Maybe could include n_ter in the object as well as computation_mode
# Computation log only displays if n_rollouts = 1

struct RolloutResult
    assignment::Array{Int64}
    expected_reward::Float64
    choice::Array{Float64}
    computation_log::Array{Int64}
end

function change_costs(p::Problem,combo::Array{Int64,1},sale_reduction::Float64,fun_input)
    for salei in combo
        p.cost[salei] = fun_input(p.cost[salei],sale_reduction)
    end
    return p
end

function loop_rollouts(p::Problem,π::Policy,assignment::Array{Int64,1};n_rollouts=100)
    total_reward = 0
    choice_counts = zeros(p.prm.n_gamble)
    computation_log = []
    for rollouti in 1:n_rollouts
        curr_rollout = rollout(π,p)
        curr_choice = curr_rollout.choice
        choice_counts[curr_choice] +=1
        total_reward += curr_rollout.assistant_expected_reward
        if rollouti == n_rollouts && n_rollouts == 1
            computation_log = curr_rollout.computation_log
        end
    end
    choice_probabilities = choice_counts / n_rollouts
    task_ev = total_reward / n_rollouts
    looped_rollout = RolloutResult(assignment,task_ev,choice_probabilities,computation_log)
    return looped_rollout
end

function sale_number(n_sale::Int,p::Problem,π::Policy,z::Number;n_rollouts=100,print_output=false)
    return_vector=[RolloutResult]
    expected_reward = -Inf
    num_combinations = length(collect(combinations(p.matrix,n_sale)))
    sale_reduction = z/n_sale
    if sale_reduction>minimum(p.cost)
        error("The budget/n_sale must be less than or equal to the min cost")
    end
    for comboi in collect(combinations(Vector(1:length(p.matrix)),n_sale))
        p = change_costs(p,comboi,sale_reduction,-) # put items on sale
        # Order of computations
        rollout_output = loop_rollouts(p,π,comboi,n_rollouts = n_rollouts)
        if print_output
            println("Current Combination: $comboi ... Current combo total: $num_combinations")
        end
        if rollout_output.expected_reward > expected_reward
            return_vector = [rollout_output]
            expected_reward = rollout_output.expected_reward
        elseif rollout_output.expected_reward == expected_reward
            push!(return_vector,rollout_output)
        end
        p = change_costs(p,comboi,sale_reduction,+)
    end
    return return_vector
end


function optimal_allocation(p::Problem,π::Policy,z::Number;print_output=false,n_min=1,n_max=length(p.matrix),n_rollouts=100)
    # z is the budget
    if ~(n_min in range(1,length(p.matrix))) || ~(n_max in range(1,length(p.matrix)))
        error("n_min and n_max must be between 1 and the length of the problem matrix")
    end
    expected_reward = -Inf
    return_vector=[RolloutResult]
    for n_salei in n_min:n_max
        curr_sale_rollout = sale_number(n_salei,p,π,z,print_output=print_output,n_rollouts=n_rollouts)
        if curr_sale_rollout[1].expected_reward > expected_reward
            return_vector = curr_sale_rollout
            expected_reward = curr_sale_rollout[1].expected_reward
        elseif curr_sale_rollout[1].expected_reward == expected_reward
            return_vector = vcat(return_vector,curr_sale_rollout)
        end
    end
    return return_vector
end
