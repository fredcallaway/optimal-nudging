### --- Script for running the power analysis for our pre-registered defaults experiment --- ###
### --- Prereg at https://osf.io/nep97 --- ###

# Load packages and helper functions
library(dplyr)
source('utils.R')

# Function to get expected p values for models 1, 2 and 3 defaults for specific sample size
hypothesis_tests = function(sim_data,sample_size,num_sims){
  print(paste('Getting power for',sample_size,'trials'))
  results = vector()
  for (i in 1:num_sims){
    curr_sample = balanced_sample(sim_data,sample_size)
    
    # H1: tTe probability of accepting the recommendation is greater than chance.
    h1 = prop.test(x = sum(curr_sample$choose_suggested), n = nrow(curr_sample), p = 1/6, correct = FALSE)
    h1_significant = h1$estimate>0.2 & h1$p.value/2<0.05

    # Fit model for H2 and H3
    model2_results = glm(choose_suggested~many_features+after,data=curr_sample,family='binomial') %>%
      summary() %>%
      coef()
    
    # H2: Increasing the number of features increases the probability of accepting the suggestion.
    h2_significant = model2_results[,1][2] >0 & model2_results[,4][2]/2<0.05
    
    # H3 The probability of accepting the recommendation is higher for early suggestions
    h3_significant = model2_results[,1][3] < 0 & model2_results[,4][3]/2<0.05
    
    # H4: The effect of features on probability of accepting the recommendation is larger for early suggestions.
    model3_results = glm(choose_suggested~many_features*after,data=curr_sample,family='binomial') %>%
      summary() %>%
      coef()
    h4_significant = model3_results[,1][4]<0 & model3_results[,4][4]/2<0.05

    # Save
    test_results = data.frame(h1_significant,h2_significant,h3_significant,h4_significant)
    test_results$power = rowSums(test_results) == ncol(test_results)
    results = rbind(test_results,results)
  }
  return(results)
}

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- ANALYSIS -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

supersize_sim_data = '../../model/results/supersize_sims.csv' %>%
  read.csv() %>%
  subset(reveal_cost==2) %>%
  subset(naive == T) %>%
  mutate(
    many_features = n_feature>4,
    metalevel_reward = payoff - decision_cost,
    # Make a column that determines the unique feature combination
    balancer_id = paste0(many_features,after)
  )

# Power will be estimated for every entry in num_participants
num_participants = c(400)
# Every participant does 20 problems
num_trials = num_participants * 20
# The number of times to simulate the experiment for every participant number
num_sims = 1000

# Get power results
power = power_analysis(supersize_sim_data,num_trials,num_sims)
print(power)