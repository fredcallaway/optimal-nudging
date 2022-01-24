### --- Script for running the power analysis for two optimal nudging experiments --- ###
### --- Belief modification prereg at https://osf.io/cs5kx --- ###
### --- Costs reduction prereg at https://osf.io/gspr4 --- ###

### NOTE: -- -- -- -- -- 
### The initial power analysis for the belief modification experiment
### used an incorrect simulation dataset to esimate the power.
### Thus, actual power is lower than what we reported in our pre-registration

# Load packages and helper functions
library(dplyr)
source("utils.R")

# Function to get expected p values for models 1, 2 and 3 defaults for specific sample size
hypothesis_tests = function(sim_data,sample_size,num_simulations){
  print(paste('Getting power for',sample_size,'trials'))
  results = data.frame()
  for (i in 1:num_simulations){
    curr_sample = balanced_sample(sim_data,sample_size)
    
    # Fit model for H1-H2
    model1_results = lm(metalevel_reward ~ name_factor,data=curr_sample) %>% summary() %>% coef()
    
    # H1: Metalevel higher on optimal trials than random trials
    h1_significant = model1_results[,1][2] < 0 & model1_results[,4][2]/2 < 0.05
    
    # H2: Metalevel higher on optimal trials than extreme trials
    h2_significant = model1_results[,1][3] < 0 & model1_results[,4][3]/2 < 0.05
    
    # Fit model for H3-H4
    model2_results = lm(payoff ~ name_factor,data=curr_sample) %>% summary() %>% coef()
    
    # H3: Action utility higher on optimal trials than random trials
    h3_significant = model2_results[,1][2] < 0 & model2_results[,4][2]/2 < 0.05
    
    # H4: Action utility higher on optimal trials than extreme trials
    h4_significant = model2_results[,1][3] < 0 & model2_results[,4][3]/2 < 0.05
    
    # Fit model for H5-H6
    model3_results = lm(decision_cost ~ name_factor,data=curr_sample) %>% summary() %>% coef()
    
    # H5: Click cost lower on optimal trials than random trials
    h5_significant = model3_results[,1][2] > 0 & model3_results[,4][2]/2 < 0.05
    
    # H6: Click cost lower on optimal trials than extreme trials
    h6_significant = model3_results[,1][3] > 0 & model3_results[,4][3]/2 < 0.05
    
    curr_results = data.frame(h1_significant,h2_significant,
                              h3_significant,h4_significant,h5_significant,h6_significant)
    # Are all tests significant?
    curr_results$power = rowSums(curr_results) == ncol(curr_results)
    results = rbind(results,curr_results)
  }
  return(results)
}

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- ANALYSIS -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

# belief_modification is T/F based on whether a
# power analysis should be run on the belief modification or 
# cost reduction simulations (experiments 4 and 5, respectively)
belief_modification = T

# Power will be estimated for every entry in num_participants
num_participants = c(250)
# Every participant does 30 problems
num_trials = num_participants * 30
# The number of times to simulate the experiment for every participant number
num_sims = 100

fname = if (belief_modification) 'belief_modification_sims' else 'cost_reduction_sims'
fpath = paste0('../../model/results/',fname,'.csv')

optimal_sim_data = fpath %>%
  read.csv() %>%
  subset(nudge_type !='none') %>%
  mutate(
    metalevel_reward = payoff-decision_cost,
    name = as.character(nudge_type),
    name_factor = factor(name, levels = c("greedy", "random", "extreme")),
    # Make a column that determines the unique feature combination
    balancer_id = name
  )

power = power_analysis(optimal_sim_data,num_trials,num_sims)
print(power)