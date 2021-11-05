library(dplyr)

# Load data
experiment_data = read.csv('../../data/final_experiments_data/traffic_light_data.csv')
experiment_data$is_practice = experiment_data$is_practice == "True"
experiment_data$is_control = experiment_data$is_control == 'True'
# Get only test trials
experiment_test = subset(experiment_data,is_practice==F)

# H1: Revealed values higher 
experiment_data$is_control == 'True'

experiment_data$is_control == 'True'

t.test(highlight_num_reveals ~ is_control, data = experiment_test, alternative='greater')

# H2: Highlight value
t.test(highlight_value ~ is_control, data=experiment_test, alternative='greater')

# H3: Probability of maximizing the highlighted_option
nudge_trials = subset(experiment_test,is_control==F)
control_trials = subset(experiment_test,is_control==T)
prop.test(
  x=c(sum(nudge_trials$maximized_highlighted_option),sum(control_trials$maximized_highlighted_option)),
  n = c(nrow(nudge_trials),nrow(control_trials)),
  alternative='greater')

# Get means for reporting
summary_data = data.frame(experiment_test  %>%
  group_by(is_control) %>%
  summarize(
    mean_highlight_num_reveals = round(mean(highlight_num_reveals),2),
    mean_highlight_value = round(mean(highlight_value),2),
    mean_maximized_highlighted_option = round(mean(maximized_highlighted_option),2)
  )
)

print(summary_data)