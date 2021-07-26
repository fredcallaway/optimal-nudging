library(dplyr)
library(lme4)

# Load and clean data
experiment_data = read.csv('../../data/final_experiments_data/default_data.csv')
experiment_data$is_practice = experiment_data$is_practice == "True"
experiment_data$many_options = experiment_data$og_baskets == 5
experiment_data$many_features = experiment_data$num_features == 5
experiment_data$revealed_no_values = experiment_data$uncovered_values == '[]'
experiment_data$points_metalevel_reward = experiment_data$net_earnings * 2000
experiment_data$chose_nudge = experiment_data$chose_nudge == "True"

# Get test data for analysis
experiment_test = subset(experiment_data,is_practice==F)

# Show model 2 with random effects lead to singular fits - will omit them in all models for consistency
mixed_model_2 = glmer(chose_nudge ~ trial_nudge * (many_options + many_features + weights_deviation) +
                        (1 | participant_id),data = experiment_test,family='binomial')

# Note: We define the relative probability of selecting the default as the difference in probability 
# between choosing the option that is best under uniform prize values 
# (i.e., the option that would be the default if a default is presented) on trials where this option 
# is presented as the default vs. trials with no default option.

# H1: Relative probability of selecting the default is positive
model_1 = glm(chose_nudge ~ trial_nudge,data = experiment_test,family='binomial')
print(summary(model_1))

# H2: The relative probability of selecting the default will be higher wheen there are many options
# H3: The relative probability of selecting the default will be higher when there are many features
# H4: The relative probability fo selecting the default will be lower on trials with higher weights deviation 
model_2 = glm(chose_nudge ~ trial_nudge * (many_options + many_features + weights_deviation),data=experiment_test,family='binomial')
print(summary(model_2))

# H5: Metalevel reward will be higher when the default is shown
print(data.frame(experiment_test %>% group_by(trial_nudge) %>% summarize(mean_metalevel_reward = mean(points_metalevel_reward))))
model_3 = lm(points_metalevel_reward ~ trial_nudge,data=experiment_test)
print(summary(model_3))

# H6: Trials with higher weight deviation will benefit have less benefit from default nudges
model_3 = lm(points_metalevel_reward ~ trial_nudge * weights_deviation,data=experiment_test)
print(summary(model_3))

# H7: The relative probability of selecting the default will be positive on trials where participants revealed at least one value
revealed_values = subset(experiment_test,revealed_no_values==F)
print(data.frame(revealed_values %>% group_by(trial_nudge) %>% summarize(mean_revealed_nudge = mean(chose_nudge))))
model_4 = glm(chose_nudge ~ trial_nudge, data=revealed_values,family='binomial')
summary(model_4)
