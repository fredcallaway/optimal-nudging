include("feature_modification.jl")
using Combinatorics

# metagreedy policy
pol = Policy([0,1,0,0,0])

# budget - make equal to the cost for now
const budget = 0.1


# set up problem
prm = Params(n_gamble=5,n_outcome=3,cost=0.1)
problem = Problem(prm)

# Optimally assign a budget to maximize metalevel reward
# n_min is the minimum number of items you want to try assigning the budget to
# n_max is the same, but the maximum
# By default these range between 1 and length(problem.matrix)
# For a given sale number, the budget is divided equally among these items

# The function below returns a RolloutResult struct
# if n_rollouts is 1, it also returns a computation log, otherwise this is an empty array
budget_assignment = optimal_allocation(problem,pol,budget,print_output=true,n_max=2,n_rollouts=10)

#
