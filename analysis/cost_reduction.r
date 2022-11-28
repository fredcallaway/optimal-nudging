source("base.r")
EXCLUDE = FALSE
# path = paste0("stats/reduction", if (EXCLUDE) "" else "-full")
path = "stats/reduction"

# %% ==================== Load data ====================

human_raw = read_csv('../data/experiments/reported_experiments/optimal_nudging_changing_costs_data.csv', col_types = cols())
model_raw = read_csv('../model/results/cost_reduction_sims.csv', col_types = cols())

# %% --------
human = human_raw %>%
    filter(!is_practice) %>% 
    transmute(
        nudge_type,
        num_values_revealed = map_int(uncovered_values, ~ length(fromJSON(.x))),
        participant_id = as.character(participant_id),
        n_option = num_baskets,
        n_feature = num_features,
        # reveal_cost = cost,
        weight_dev = weights_deviation,
        weights = map(weights, fromJSON),
        decision_cost = points_click_cost,
        payoff = gross_earnings * 3000,
        total_points = net_earnings * 3000,
    ) # %>% apply_exclusion(TRUE)

report_exclusion(path, human_raw, human)

model = model_raw %>% 
    select(-problem_id) %>% 
    filter(nudge_type != "none") %>% 
    mutate(
        participant_id = "model",
        total_points = payoff - decision_cost
    )

df = bind_rows(human, model) %>% mutate(
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature),
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
    nudge = recode_factor(nudge_type, 'random'='Random', 'extreme'='Extreme', 'greedy'='Optimal'),
    cond = glue("{n_option} Options {n_feature} Features")
)
human = filter(df, model=="Human")
model = filter(df, model=="Model")

# %% ==================== Plot ====================

random_payoff = 150
maximum_payoff = 183.63861

p1 = df %>% 
    ggplot(aes(nudge, total_points, group=0)) +
        stat_summary(fun=mean, geom="line") +
        point_and_error + 
        geom_hline(yintercept=c(maximum_payoff), linetype="dashed") +
        coord_cartesian(xlim=c(NULL), ylim=c(random_payoff, maximum_payoff)) +
        facet_rep_grid(~model) +
        # theme(axis.title.x=element_blank()) + 
        labs(x="Nudge Type", y="Total Points")

savefig("cost_reduction", 7, 3)

p2 = human_raw %>%
  filter(!is_practice) %>%
  mutate(
    test_trial = trial_num - 2,
    total_points = net_earnings * 3000,
    Nudge = paste(toupper(substr(nudge_type, 1, 1)), substr(nudge_type, 2, nchar(nudge_type)), sep="")
  ) %>%
  group_by(test_trial,Nudge) %>%
  summarize(average_total_points = mean(total_points)) %>%
  ggplot(aes(x=test_trial,y=average_total_points,color=Nudge,shape=Nudge)) +
  geom_smooth(alpha=0.2) +
  stat_summary_bin(fun.data=mean_se, bins=5) +
  scale_x_continuous(limits=c(0,31)) +
  labs(x="Trial number", y="Total points")

savefig("cost_reduction_learning_curves", 4, 3)

# %% ==================== Stats ====================
human$nudge = factor(human$nudge, levels = c("Optimal", "Extreme", "Random"))

write_effect = function(var, fmt="{val:.1}") {
    svar = substitute(var)
    human %>% 
        group_by(nudge) %>% 
        summarise(val=mean({{var}})) %>%
        rowwise() %>% group_walk(~ with(.x, 
            write_tex("{path}/{svar}/mean/{nudge}", fmt)
        ))
    human %>% 
        tidylm(nudge, {{var}}) %>% 
        write_model("{path}/{svar}/model")
}

write_effect(total_points)
write_effect(payoff)
write_effect(decision_cost)




# %% ==================== Explore ====================
quit()  # don't run below in script
# %% --------


df %>% 
    pivot_longer(c(total_points, payoff), names_to="name", values_to="value", names_prefix="") %>% 
    ggplot(aes(nudge, value, group=name)) +
        stat_summary(fun=mean, geom="line") +
        point_and_error + 
        # feature_colors +
        facet_rep_grid(~model) +
        geom_hline(yintercept=c(maximum_payoff), linetype="dashed")
        coord_cartesian(xlim=c(NULL), ylim=c(random_payoff, maximum_payoff))
        # scale_linetype_manual(values=c("dotted", "solid")) +
        # labs(linetype="Options", x='Nudge', y='Prob Choose Default')
savefig("belief_modification", 7, 3)

# %% --------

df %>%
    ggplot(aes(nudge, decision_cost, group=0)) +
        stat_summary(fun=mean, geom="line") +
        point_and_error + 
        # feature_colors +
        facet_rep_grid(~model) 
        # scale_linetype_manual(values=c("dotted", "solid")) +
        # labs(linetype="Options", x='Nudge', y='Prob Choose Default')
savefig("tmp", 7, 3)


#

df %>%
    ggplot(aes(weight_dev, payoff, color=nudge, fill=nudge)) + 
    geom_smooth(alpha=0.2) +
    nudge_colors +
    facet_rep_wrap(~model) +
    stat_summary_bin(fun.data=mean_se, bins=5)

fig("tmp", 7, 3)

# %% --------
df %>%
    mutate(high_dev = weight_dev > median(df$weight_dev)) %>% 
    ggplot(aes(nudge, payoff, color=high_dev, group=interaction(n_option, n_feature, high_dev))) + 
    stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
    point_and_error + 
    # nudge_colors +
    # facet_rep_wrap(~model)
    facet_rep_grid(interaction(n_feature, n_option) ~ model, scales="free_y")


fig("tmp", 7, 7)
# %% --------
df %>%
    mutate(high_dev = weight_dev > median(df$weight_dev)) %>% 
    ggplot(aes(nudge, payoff, color=high_dev, group=high_dev)) + 
    stat_summary(fun=mean, geom="line") +
    point_and_error + 
    # nudge_colors +
    # facet_rep_wrap(~model)
    facet_rep_grid(interaction(n_feature, n_option) ~ model, scales="free_y")


fig("tmp", 7, 7)

# %% --------
df %>% 
    mutate(high_dev = 
        factor(if_else(weight_dev > median(df$weight_dev), "Idiosyncratic", "Typical"))
    ) %>% 
    group_by(model, n_option, n_feature, high_dev, participant_id, nudge) %>% 
    summarise(val=mean(total_points)) %>% 
    pivot_wider(names_from=nudge, values_from=val) %>% 
    transmute(benefit = Present - Absent) %>% 
    ggplot(aes(high_dev, benefit, color=n_feature, group=interaction(n_option,n_feature))) +
        geom_hline(yintercept=0) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        point_and_error + 
        feature_colors +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid")) +
        labs(x="Preference Type", y="Effect of Nudge on Earnings")

fig("tmp", 7, 3)

# %% --------
df %>% 
    mutate(high_dev = 
        factor(if_else(weight_dev > median(df$weight_dev), "Idiosyncratic", "Typical"))
    ) %>% 
    group_by(model, high_dev, participant_id, nudge) %>% 
    summarise(val=mean(total_points)) %>% 
    pivot_wider(names_from=nudge, values_from=val) %>% 
    transmute(benefit = Present - Absent) %>% 
    ggplot(aes(high_dev, benefit, group=1)) +
        stat_summary(fun=mean, geom="line") +
        point_and_error + 
        feature_colors +
        coord_cartesian(xlim=c(NULL), ylim=c(0, NA)) +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid")) +
        labs(x="Preference Type", y="Effect of Nudge on Earnings")

fig("tmp", 7, 3)

# %% --------

df %>% 
    ggplot(aes(weight_dev, y=..prop..)) + 
    geom_bar()+
    facet_rep_grid(interaction(n_feature, n_option) ~ model, scales="free_y")
fig("tmp", 7, 7)


# %% --------
df %>% 
    ggplot(aes(n_feature, chose_nudge, color=nudge, group=interaction(nudge,n_option))) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        point_and_error + 
        nudge_colors +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid"))

savefig("tmp", 7, 3)

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
