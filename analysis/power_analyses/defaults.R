### --- Script for running the power analysis for our pre-registered defaults experiment --- ###
### --- Prereg at https://osf.io/4fcny --- ###

### NOTE: -- -- -- -- -- 
### Unlike other power analyses, this script uses two-tailed tests
### as we were planning on including participant random effects

# Load packages and helper functions
library(ggplot2)
source("utils.R")

# Function to get power for a specific sample size
hypothesis_tests = function(sim_data,sample_size,num_inner_samples){
  print(paste('Getting power for',sample_size,'trials'))
  results = data.frame()
  for (i in 1:num_inner_samples){
    # Get a sample that matches the constraints in the experiment
    curr_sample = balanced_sample(sim_data,sample_size)
    
    # H0: Relative probability of selecting the default is positive
    model0_outcomes = glm(choose_default~nudge,data=curr_sample,family='binomial') %>% summary() %>% coef()
    h0_significant = model0_outcomes[,4]<0.05 & model0_outcomes[,1] > 0

    # Fit a model for testing H1-H3
    model1_outcomes = glm(choose_default ~ nudge * (many_options+many_features+weight_dev),
                          data=curr_sample,family='binomial') %>% summary() %>% coef()
    model1_p_vals = model1_outcomes[,4]
    model1_coefs = model1_outcomes[,1]
    
    # H1: Higher chance of selecting the default when there are more options
    h1_significant = model1_p_vals[6] <0.05 & model1_coefs[6] > 0
    # H2: Higher chance of selecting the default when there are more features
    h2_significant = model1_p_vals[7]<0.05 & model1_coefs[7] > 0
    # H3: Lower chance of selecting the default when idiosyncrasy is high
    h3_significant = model1_p_vals[8]<0.05 & model1_coefs[8] < 0

    # H4: Higher metalevel reward on default trials
    model2_outcomes = lm(metalevel_reward ~ nudge,data=curr_sample) %>% summary() %>% coef()
    h4_significant = model2_outcomes[,4][2]<0.05 & model2_outcomes[,1][2]>0
    
    # H5: Benefit of defaults less when idiosyncrasy is high
    model3_outcomes = lm(metalevel_reward ~ nudge*weight_dev,data=curr_sample) %>% summary() %>% coef()
    h5_significant = model3_outcomes[,4][4]<0.05 & model3_outcomes[,1][4]<0
    
    # H6: Relative probability of selecting the default positive on trials where participants revealed a value    
    clicked_data = subset(curr_sample,decision_cost>0)
    model4_outcomes = glm(choose_default ~ nudge,data=clicked_data,family='binomial') %>% summary() %>% coef()
    h6_significant = model4_outcomes[,4][2]<0.05 & model4_outcomes[,1][2]>0
  
    curr_results = data.frame(h0_significant,h1_significant,h2_significant,
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

# Power will be estimated for every entry in num_participants
num_participants = c(400)
# Every participant does 32 problems
num_trials = num_participants * 32
# The number of times to simulate the experiment for every participant number
num_sims = 1000

# Load and prepare data
defaults_sim_data = '../../model/results/default_sims.csv' %>%
  read.csv() %>%
  subset(reveal_cost==2) %>%
  mutate(
    many_options = n_option>4,
    many_features = n_feature>4,
    metalevel_reward = payoff-decision_cost,
    # Make a column that determines the unique feature combination
    balancer_id = paste0(many_options,many_features,nudge)
  )

# Get power results
power_df = power_analysis(defaults_sim_data,num_trials,num_sims)
print(power_df)