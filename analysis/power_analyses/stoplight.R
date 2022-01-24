### --- Script for running the power analysis for our pre-registered stoplight experiment --- ###
### --- Prereg at https://osf.io/8pyex/ --- ###

# Load packages
library(dplyr)

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- HELPER FUNCTIONS -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

# Runs a power analysis given:
# A vector of number of simulations for each unique parameter combination (num_sims_per_combo)
# A vector of different numbers of participants to simulate (sample sizes)
# A int for the number of problems each participant should do (num_problems)
stoplight_power_analysis = function(num_sims_per_combo,sample_sizes,num_problems){
  # Load sim data
  simulation_data = read.csv('../../model/results/attention_sims.csv')
  simulation_data$is_control = simulation_data$nudge=='absent'
  
  # Subset into control and nudge data
  control_data = subset(simulation_data,is_control==T)
  nudge_data = subset(simulation_data,is_control==F)

  results = data.frame()
  # Loop through num participants, simulating the experiment num_sims_per_combo each time
  for (num_participants_i in sample_sizes){
    print(paste("Running",num_participants_i,'participants'))
    for (sim_i in 1:num_sims_per_combo){
      
      # On average, half the participants will have trials with even nudge highlight weights, and half odd
      even_nudge_participants = sum(rbinom(num_participants_i,1,0.5))
      odd_nudge_participants = num_participants_i - even_nudge_participants
      
      # Sample trials with highlight weight constraint
      nudge_trials =  sample_trials(nudge_data,even_nudge_participants,odd_nudge_participants)
      
      # Sample control trials with highlight weight constraint
      # Note that participants with even nudge weights have odd control weights, and vice versa
      control_trials =  sample_trials(control_data,odd_nudge_participants, even_nudge_participants)
      
      # Run analyses on sampled data for this experiment
      experiment_outcomes = hypothesis_tests(nudge_trials,control_trials,num_problems,num_participants_i,sim_i)
      
      # Bind to previously simulated data
      results = rbind(results,experiment_outcomes)
    }
  }
  return(results)
}

# Function that makes sure that for each highlight weight in intput_data,
# you sample num_participants problems with replacement
# This replicates the weight randomization procedure used in the experiment
sample_trials = function(input_data,num_participants_even_weights,num_participants_odd_weights){
  return(
    input_data %>%
      group_by(weight_highlight) %>%
      sample_n(
        if(unique(weight_highlight) %% 2 == 0) num_participants_even_weights else num_participants_odd_weights,
        replace=T) %>%
      data.frame()
  )
}

# Runs all pre-registered hypothesis tests on sampled sim data 
hypothesis_tests = function(nudge_trials,control_trials,num_problems,num_participants,simulation){
  # H1: Revealed values higher on nudge trials
  revealed_values_p_value = t.test(nudge_trials$n_click_highlight,
                                   control_trials$n_click_highlight,
                                   alternative='greater')$p.value
  
  # H2: Highlight value higher on nudge trials
  highlight_value_p_value = t.test(nudge_trials$highlight_value,
                                   control_trials$highlight_value,
                                   alternative='greater')$p.value
  
  # H3: Probability of maximizing the highlighted option higher on nudge trials
  maximized_highlight_p_value = prop.test(
    x=c(sum(nudge_trials$max_highlight_value),sum(control_trials$max_highlight_value)),
    n = c(nrow(nudge_trials),nrow(control_trials)),
    alternative='greater')$p.value
  
  # Bind all p values together along with binary indicators of signifiance
  outcomes = data.frame(
    num_participants,
    simulation,
    num_problems,
    revealed_values_p_value,
    highlight_value_p_value,
    maximized_highlight_p_value,
    revealed_values_significant = revealed_values_p_value<0.05,
    highlight_value_significant = highlight_value_p_value<0.05,
    maximized_highlight_significant = maximized_highlight_p_value<0.05
  )
  
  # Add an indicator of whether all hypothesis tests were significant
  outcomes$all_significant = outcomes %>%
    select(contains("significant")) %>%
    rowSums() == 3
  
  return(outcomes)
}

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- ANALYSIS -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

# Set parameters
num_sims = 100
# Number of participants
sample_sizes = c(100,150,200,250,300)
num_problems_per_participant = 28

# Get raw power data given num_sims, sample_sizes, and num_problems
power = num_sims %>% 
  stoplight_power_analysis(sample_sizes,num_problems_per_participant) %>%
  # Group power data by number of participants and summarize
  group_by(num_participants) %>%
  summarize(
    revealed_values_power = mean(revealed_values_significant),
    highlight_value_power = mean(highlight_value_significant),
    maximized_highlight_power = mean(maximized_highlight_significant),
    power = mean(all_significant)
  ) %>%
  data.frame()

# Investigate power data
print(power)