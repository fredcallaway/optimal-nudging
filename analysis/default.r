source("base.r")

# %% ==================== Load data ====================

human_raw = read_csv('../data/final_experiments_data/default_data.csv', col_types = cols())
model_raw = read_csv('../model/results/default_sims.csv', col_types = cols())

# %% --------

human = human_raw %>%
    filter(!is_practice) %>% 
    transmute(
        num_values_revealed = map_int(uncovered_values, ~ length(fromJSON(.x))),
        participant_id = as.character(participant_id),
        n_option = og_baskets,
        n_feature = num_features,
        reveal_cost = cost,
        nudge = trial_nudge == 'default',
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = gross_earnings,        
        chose_nudge = as.integer(chose_nudge),
    ) %>% apply_exclusion(!nudge)

model = model_raw %>% 
    filter(reveal_cost == only(unique(human$reveal_cost))) %>% 
    select(-n_click_default) %>% 
    mutate(
        participant_id = "model",
        chose_nudge = choose_default
    )

df = bind_rows(human, model) %>% mutate(
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature),
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
    nudge = factor(nudge, levels=c(0,1), labels=c("Absent", "Present")),
    cond = glue("{n_option} Options {n_feature} Features")
)

# %% ==================== Plot ====================

df %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=cond)) +
        stat_summary(fun.data=mean_se, geom="pointrange", size=.2) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        facet_rep_grid(~model) + 
        feature_colors +
        scale_linetype_manual(values=c("dotted", "solid")) +
        labs(linetype="Options", x='Nudge', y='Prob Choose Default')

savefig("default", 7, 3)

# %% ==================== Explore ====================
quit()  # don't run below in script
# %% --------

human %>% 
    mutate(nudge = factor(nudge, levels=c(F,T), labels=c("Absent", "Present"))) %>% 
    mutate(no_click = num_values_revealed == 0) %>%
    group_by(nudge,participant_id) %>% 
    summarise(n=sum(no_click)) %>% 
    ggplot(aes(n, fill=nudge)) + geom_bar(position="dodge") + nudge_colors +labs(x="Number of trials with no reveals")
fig(w=6)

# %% --------

# df %>% mutate(cond = glue("{n_option} Options {n_feature} Features")) %>% 
#     ggplot(aes(nudge, chose_nudge, color=cond, group=cond)) +
#         stat_summary(fun.data=mean_se, size=.2) +
#         stat_summary(fun.data=mean_se, geom="line") +
#         scale_colour_manual(values=c(
#             "gray",
#             "dodgerblue",
#             "red2",
#             "purple2"
#         ), aesthetics=c("fill", "colour"), name="") +
#     facet_wrap(~model)

# fig(w=7, h=3)
