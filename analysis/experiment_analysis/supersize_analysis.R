library(dplyr)

# Load and clean data
experiment_data = read.csv('../data/final_experiments_data/supersize_data.csv')
experiment_data$chose_nudge = experiment_data$chose_nudge == "True"
experiment_data$is_practice = experiment_data$is_practice == "True"
experiment_data$num_features = as.integer(experiment_data$num_features)
experiment_data$num_options = as.integer(experiment_data$og_baskets)
experiment_data$many_features = experiment_data$num_features==5
experiment_data$many_options = experiment_data$num_options==5

# Get trials with a nudge
nudge_test = subset(experiment_data,is_practice==F & trial_nudge!='control')
nudge_test$after = nudge_test$trial_nudge=='post-supersize'

# H1: Probability of accepting the nudge greater than chance
h1 = prop.test(x = sum(nudge_test$chose_nudge), n = nrow(nudge_test), p = 1/6, correct = FALSE,alternative='greater')
print(h1)

# H2: Participants will choose the nudge more when there are many features
# H3: Participants will accept the nudge more when the suggestion is given early
model_1 = glm(chose_nudge ~ many_features+after,data=nudge_test,family='binomial')
print(summary(model_1))

# H4: The effect of many features on probability of accepting the nudge will be higher for early suggestions
# IE, the difference between chose nudge when many features == T between after T/F is larger than the difference
# between chose nudge when many features == T between after 
model_2 = glm(chose_nudge ~ many_features*after,data=nudge_test,family='binomial')
print(summary(model_2))

# Get condition means
summarized_nudge_test= nudge_test %>% 
  group_by(num_features,trial_nudge)  %>% 
  summarize(mean_chose_suggestion = mean(chose_nudge))

print(summarized_nudge_test)