library(dplyr)
library(ggplot2)

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- FUNCTIONS -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

# Runs a power analysis given:
# A vector of number of simulations for each unique parameter combination (num_sims_per_combo)
# A vector of different numbers of participants to simulate (sample sizes)
# A int for the number of problems each participant should do (num_problems)
power_analysis = function(num_sims_per_combo,sample_sizes,num_problems){
  # Load sim data
  simulation_data = read.csv('../../model_simulation_data/stoplight_sims.csv')
  simulation_data$is_control = ifelse(simulation_data$nudge=='absent',T,F)
  
  # Subset into control and nudge data
  control_data = subset(simulation_data,is_control==T)
  nudge_data = subset(simulation_data,is_control==F)
  
  # Loop through num participants, simulating the experiment num_sims_per_combo each time
  first_iteration=T
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
      experiment_outcomes = run_hypothesis_tests(nudge_trials,control_trials,num_problems,num_participants_i,sim_i)
      
      # Bind to previously simulated data
      if(first_iteration){
        all_df = experiment_outcomes
        first_iteration=F
      } else {
        all_df = rbind(all_df,experiment_outcomes)
      }
      
    }
  }
  return(all_df)
}

# Function that makes sure that for each highlight weight in intput_data,
# you sample num_participants problems with replacement
# This replicates the weight randomization procedure used in the experiment
sample_trials = function(input_data,num_participants_even_weights,num_participants_odd_weights){
  sampled_data = input_data %>%
    group_by(highlight_weight) %>%
    sample_n(
      if(unique(highlight_weight) %% 2 == 0) num_participants_even_weights else num_participants_odd_weights,
      replace=T)
  return(data.frame(sampled_data))
}

# Runs all pre-registered hypothesis tests on sampled sim data 
run_hypothesis_tests = function(nudge_trials,control_trials,num_problems,num_participants,simulation_num){
  # H1: Revealed values higher on nudge trials
  revealed_values_p_value = t.test(nudge_trials$num_highlight_reveals,
                                   control_trials$num_highlight_reveals,
                                   alternative='greater')$p.value
  
  # H2: Highlight value higher on nudge trials
  highlight_value_p_value = t.test(nudge_trials$highlight_value_check,
                                   control_trials$highlight_value_check,
                                   alternative='greater')$p.value
  
  # H3: Probability of maximizing the highlighted option higher on nudge trials
  maximized_highlight_p_value = prop.test(
    x=c(sum(nudge_trials$max_highlight_value),sum(control_trials$max_highlight_value)),
    n = c(nrow(nudge_trials),nrow(control_trials)),
    alternative='greater')$p.value
  
  # Bind all p values together along with binary indicators of signifiance
  curr_outcomes = data.frame(
    revealed_values_p_value = revealed_values_p_value,
    highlight_value_p_value = highlight_value_p_value,
    maximized_highlight_p_value = maximized_highlight_p_value,
    revealed_values_significant = revealed_values_p_value<0.05,
    highlight_value_significant = highlight_value_p_value<0.05,
    maximized_highlight_significant = maximized_highlight_p_value<0.05,
    num_participants = num_participants,
    simulation = simulation_num,
    num_problems = num_problems
  )
  
  # Add an indicator of whether all hypothesis tests were significant
  curr_outcomes$all_significant = curr_outcomes$revealed_values_significant + 
    curr_outcomes$highlight_value_significant + 
    curr_outcomes$maximized_highlight_significant == 3
  
  return(curr_outcomes)
}

# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- ANALYSIS -- -- -- --
# -- -- -- -- -- -- -- -- -- -- -- 
# -- -- -- -- -- -- -- -- -- -- -- 

# Set parameters
num_sims = 1000
sample_sizes = c(100,150,200,250,300)
num_problems = 28

# Get raw power data given num_sims, sample_sizes, and num_problems
sim_power_data = power_analysis(num_sims,sample_sizes,num_problems)

# Group power data by number of participants for analysis and plotting
grouped_sim_power_data = data.frame(
  sim_power_data %>%
    group_by(num_participants) %>%
    summarize(
      revealed_values_power = mean(revealed_values_significant),
      highlight_value_power = mean(highlight_value_significant),
      maximized_highlight_power = mean(maximized_highlight_significant),
      power = mean(all_significant)
    )
)

# Investigate grouped power data
print(grouped_sim_power_data)

# Plot grouped power data
ggplot(grouped_sim_power_data,aes(x=num_participants,y=power)) + 
  geom_point() + geom_line() +
  xlab("Number of participants") +
  ylab("Power") +
  ggtitle("Experiment power by number of participants")
