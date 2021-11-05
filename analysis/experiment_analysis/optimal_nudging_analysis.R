library(dplyr)

# Change csv to reference relevant experiment data
# Can also be optimal_nudging_changing_initial_costs_data.csv
experiment_string = 'optimal_nudging_changing_belief_state_data.csv'

# Load data
experiment_data = read.csv(paste0('../../data/final_experiments_data/',experiment_string))
experiment_data$is_practice = as.logical(experiment_data$is_practice)
# Make greedy reference level
experiment_data$nudge_type = factor(experiment_data$nudge_type, levels = c("greedy", "extreme", "random"))

test_data = subset(experiment_data,is_practice==F)

# H1 and H2
model_1 = lm(points_metalevel_reward ~ nudge_type,data=test_data)
summary(model_1)

# H3 and H4
model_2 = lm(points_action_utility ~ nudge_type,data=test_data)
summary(model_2)

# H5 and H6
model_3 = lm(points_click_cost ~ nudge_type,data=test_data)
summary(model_3)

# Get condition means
data.frame(test_data %>% 
  group_by(nudge_type) %>% 
  summarize(
    mean_metalevel_reward = round(mean(points_metalevel_reward),2),
    mean_action_utility = round(mean(points_action_utility),2),
    mean_click_cost = round(mean(points_click_cost),2)
  )
)

c = read.csv('../../../nudging/data_analysis/calculated_bonuses/attention-experiment-13-bonuses.csv')
