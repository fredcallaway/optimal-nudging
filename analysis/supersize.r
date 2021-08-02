source("base.r")

# %% ==================== Load data ====================

human_raw = read_csv('../data/final_experiments_data/supersize.csv', col_types = cols())
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
    ) %>% apply_exclusion(nudge == "Absent")

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
    select(-c(naive, after, choose_suggested))


df = bind_rows(human, model) %>% mutate(
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature)
) %>% filter(nudge != "Absent")

# %% ==================== Plot ====================

df %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=n_feature)) +
    facet_rep_grid(~model) + 
    stat_summary(fun.data=mean_se, geom="line") +
    point_and_error +
    feature_colors +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 0.5)) + 
    labs(x="Suggestion Time", y="Prob Choose Suggestion")

savefig("supersize", 7, 3)
