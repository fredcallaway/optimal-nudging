source("base.r")
path = paste0("stats/supersize", if (EXCLUDE) "" else "-full")

# %% ==================== Load data ====================

human_raw = read_csv('../data/experiments/reported_experiments/supersize_data.csv', col_types = cols())
model_raw = read_csv('../model/results/supersize_sims.csv', col_types = cols())
# %% --------

human = human_raw %>%
    transmute(
        num_values_revealed = map_int(uncovered_values, ~ length(fromJSON(.x))),
        participant_id = as.character(participant_id),
        n_option = og_baskets,
        n_feature = num_features,
        reveal_cost = cost,
        nudge = factor(trial_nudge, levels=c("control", "pre-supersize", "post-supersize"), labels=c("Absent", "Early", "Late"), ordered=T),
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = gross_earnings,        
        chose_nudge = as.integer(chose_nudge),
        num_values_revealed = map_int(uncovered_values, ~ length(fromJSON(.x)))
    ) %>% apply_exclusion(nudge == "Absent")

report_exclusion(path, human_raw, human)

model = model_raw %>% 
    filter(
        reveal_cost == only(unique(human$reveal_cost)) &
        n_option == only(unique(human$n_option)) &
        naive == 1
    ) %>% 
    mutate(
        participant_id = "model", 
        chose_nudge = as.integer(choose_suggested),
        nudge = factor(after, levels=c(-1, 0, 1), labels=c("Absent", "Early", "Late"), ordered=T),
    ) %>% 
    rename(num_values_revealed = n_click) %>% 
    select(-c(naive, after, choose_suggested))

df = bind_rows(human, model) %>% mutate(
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature)
) %>% filter(nudge != "Absent")

# %% ==================== Plot ====================

p1 = df %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=n_feature)) +
    facet_rep_grid(~model) + 
    stat_summary(fun.data=mean_se, geom="line") +
    point_and_error +
    feature_colors +
    geom_hline(aes(yintercept = 1/6), lty="dashed") +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 0.5)) + 
    labs(x="Suggestion Time", y="Prob Choose Suggestion")

savefig("supersize", 7, 3)

## learning curves: no exclusion
p2 = human_raw %>%
  filter(!is_practice) %>%
  subset(trial_nudge != 'control') %>%
  mutate(
    Nudge = ifelse(trial_nudge == 'pre-supersize','Early suggestions','Late suggestions'),
    chose_nudge = as.integer(chose_nudge),
    test_trial = trial_num - 2,
    n_feature = as.factor(num_features)
  ) %>%
  group_by(test_trial,Nudge,n_feature) %>%
  summarize(average_choose_nudge = mean(chose_nudge)) %>%
  ggplot(aes(x=test_trial,y=average_choose_nudge,color=n_feature,group=n_feature)) +
  geom_smooth(alpha=0.2) +
  stat_summary_bin(fun.data=mean_se, bins=5) +
  scale_x_continuous(limits = c(0,31)) +
  facet_grid(cols=vars(Nudge)) +
  feature_colors +
  labs(x="Trial Number", y="Prob Choose Suggestion")

savefig("supersize_learning_curves", 7, 3)


# %% --------

df %>% 
    ggplot(aes(nudge, num_values_revealed, color=n_feature, group=n_feature)) +
    facet_rep_grid(~model) + 
    stat_summary(fun.data=mean_se, geom="line") +
    point_and_error +
    feature_colors +
    # coord_cartesian(xlim=c(NULL), ylim=c(0, 0.5)) + 
    labs(x="Suggestion Time", y="Number of Values Revealed")

savefig("tmp", 7, 3)

# %% ==================== Stats ====================

nudge_test = human %>% 
    filter(nudge != "Absent") %>% 
    mutate(
        many_options = as.integer(n_option == 5),
        many_features = as.integer(n_feature == 5),
        after = as.integer(nudge == "Late"),
    )
# H1: Probability of accepting the nudge greater than chance
nudge_test %>% 
    with(prop.test(sum(chose_nudge), n=length(chose_nudge), 
         p=1/6, correct=FALSE, alternative='greater')) %>% 
    tidy %>% 
    with(write_tex("{path}/proptest", 
        "{100*estimate:.1}\\% vs. {100/6:.1}\\%, $\\chi^2({parameter}) = {round(statistic)},\\ {pval(p.value)}$"))

# H2: Participants will choose the nudge more when there are many features
# H3: Participants will accept the nudge more when the suggestion is given early
glm(chose_nudge ~ many_features+after, data=nudge_test, family='binomial') %>% 
    write_model("{path}/choice_simple")

# H4: The effect of many features on probability of accepting the nudge will be higher for early suggestions
# IE, the difference between chose nudge when many features == T between after T/F is larger than the difference
# between chose nudge when many features == T between after 
glm(chose_nudge ~ many_features*after,data=nudge_test, family='binomial') %>% 
    write_model("{path}/choice_interaction")

# %% ====================  ====================

df %>% 
    filter(model == "Model" & n_feature==2) %>% 
    summarise(mean(payoff))


# %% --------

df %>% 
    group_by(model, nudge) %>% 
    summarise(mean(num_values_revealed==0))

df %>% 
    filter(num_values_revealed > 0) %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=n_feature)) +
    facet_rep_grid(~model) + 
    stat_summary(fun.data=mean_se, geom="line") +
    point_and_error +
    feature_colors +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 0.5)) + 
    labs(x="Suggestion Time", y="Prob Choose Suggestion")

savefig("supersize-alt", 7, 3)