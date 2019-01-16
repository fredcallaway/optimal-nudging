include("mouselab.jl")
include("matt_code.jl")
using Combinatorics
using Distances # julia pacakge for computing distances / divergences

pol = Policy([0,1,0,0,0])
prm = Params(n_gamble=4,n_outcome=3,cost=0.1)
problem = Problem(prm)

const budget = 0.01

struct PolicyDifference
    difference::Float64
    original_ev::Float64
    modified_ev::Float64
    original_choice::Array{Float64,1}
    modified_choice::Array{Float64,1}
end

# Use function loop_rollouts with assignment = 0
unmodified_policy(p::Problem,π::Policy;n_rollouts=100) = loop_rollouts(p,π,[0];n_rollouts=n_rollouts)

# Smooth probability vectors as many of them end up with all the probability on a single
# action
function smooth_probs(choices::Array{Float64,1};ϵ=0.000001)
    choices .+= ϵ
    choices ./= sum(choices)
    return choices
end

# Function that takes in a problem, assignment vector, and budget, and returns
# the modified problem
# Can have input of + or - to modify whether or the modification is adding or
# subtracting costs

function modify_MDP(p::Problem,assignment::Array{Int64,1},z::Float64,fun_input= -)
    modified_problem = deepcopy(p)
    sale_reduction = z/length(assignment)
    modified_problem = change_costs(modified_problem,assignment,sale_reduction,fun_input)
    return modified_problem
end


# Gets similarity between object-level policy in one unmodified problem p_original
# and another modified MDP p_modified
# Computed similarity with function input similarity_type (i.e., KL divergence)
# Returns both the similarity and the expected value of p_original, p_modified
function get_similarity(p_original::Problem,p_modified::Problem,
    π::Policy; similarity_fun=kl_divergence, n_rollouts=100,smooth_choices=false)
    orig_rollout = loop_rollouts(p_original,pol,[0],n_rollouts=n_rollouts)
    mod_rollout = loop_rollouts(p_modified,pol,[0],n_rollouts=n_rollouts)
    if smooth_choices
        orig_choice = smooth_probs(orig_rollout.choice)
        mod_choice = smooth_probs(mod_rollout.choice)
        orig_rollout = RolloutResultIterated(orig_rollout.assignment,
        orig_rollout.expected_reward,orig_choice)
        mod_rollout = RolloutResultIterated(mod_rollout.assignment,
        mod_rollout.expected_reward,mod_choice)
    end
    policy_difference = similarity_fun(orig_rollout.choice,mod_rollout.choice)
    curr_difference = PolicyDifference(policy_difference, orig_rollout.expected_reward,
    mod_rollout.expected_reward,orig_choice,mod_choice)
    return curr_difference
end

function optimize_with_constraint(n_sale::Int64,p::Problem,π::Policy,z::Float64;
    similarity_fun=kl_divergence,n_rollouts=100,smooth_choices=true,epsilon=0.5)
    original_mdp = p
    best_value =  loop_rollouts(p,π,[0],n_rollouts=n_rollouts).expected_reward
    best_mdp = p
    assignment = [Int64]

    for comboi in collect(combinations(Vector(1:length(p.matrix)),n_sale))
        println(comboi)
        curr_modified_p = modify_MDP(p,comboi,z,-)
        curr_similarity =  get_similarity(p,curr_modified_p,
            π; similarity_fun=similarity_fun, n_rollouts=n_rollouts,
            smooth_choices=smooth_choices)
        curr_value = curr_similarity.modified_ev
        curr_difference = curr_similarity.difference
        if curr_similarity.difference < epsilon
            if curr_value > best_value
                best_value = curr_value
                best_mdp = curr_modified_p
                assignment = comboi
            end
        end
    end

    return original_mdp, best_mdp, assignment, best_value
end

a = optimize_with_constraint(1,problem,pol,budget)
aa = rollout(pol, a[1])

b = loop_rollouts(a[1],pol,[0],n_rollouts=10000)
c = loop_rollouts(a[2],pol,[0],n_rollouts=10000)
