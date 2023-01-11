source("base.r")
path = paste0("stats/default", if (EXCLUDE) "" else "-full")

# %% ==================== Load data ====================

human_raw = read_csv('../data/experiments/reported_experiments/default_data.csv', col_types = cols())
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
        weights = map(weights, fromJSON),
        decision_cost = click_cost,
        payoff = gross_earnings * 3000,
        total_points = net_earnings * 3000,
        chose_nudge = as.integer(chose_nudge),
    ) %>% apply_exclusion(!nudge)

report_exclusion(path, human_raw, human)

model = model_raw %>% 
    filter(reveal_cost == only(unique(human$reveal_cost))) %>% 
    select(-n_click_default) %>% 
    mutate(
        participant_id = "model",
        chose_nudge = choose_default,
        total_points = payoff - decision_cost,
        num_values_revealed = decision_cost / only(unique(reveal_cost))
    )

df = bind_rows(human, model) %>% mutate(
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature),
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
    nudge = factor(nudge, levels=c(0,1), labels=c("Absent", "Present")),
    cond = glue("{n_option} Options {n_feature} Features")
)


# %% ==================== Plot ====================

p1 = df %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=cond)) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        point_and_error + 
        feature_colors +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid")) +
        labs(linetype="Options", x='Nudge', y='Prob Choose Default')
# savefig("default-choice", 7, 3)

p2 = df %>%
    ggplot(aes(weight_dev, total_points, color=nudge, fill=nudge)) + 
    geom_smooth(alpha=0.2) +
    stat_summary_bin(fun.data=mean_se, bins=5) +
    facet_rep_wrap(~model) +
    xlim(0, 40) +
    nudge_colors +
    payoff_line_lims +
    labs(x="Preference Idiosyncrasy", y="Net Earnings")

# savefig("default-utility", 7, 3)

(p1 / p2) + plot_annotation(tag_levels = 'A')
savefig("default", 7, 6)

# learning curves: no exclusion
p3 = human_raw %>%
  filter(!is_practice) %>%
  mutate(
    nudge = trial_nudge == 'default',
    chose_nudge = as.integer(chose_nudge),
    test_trial = trial_num - 2,
    nudge = ifelse(nudge,'Present','Absent')
  ) %>%
  group_by(test_trial,nudge) %>%
  summarize(average_choose_nudge = mean(chose_nudge)) %>%
  ggplot(aes(x=test_trial,y=average_choose_nudge,color=nudge)) +
  geom_smooth(alpha=0.2) +
  stat_summary_bin(fun.data=mean_se, bins=5) +
  nudge_colors +
  scale_x_continuous(limits = c(0,32)) +
  labs(x="Trial Number", y="Prob Choose Default") +
  theme(legend.position='none')

p4 = human_raw %>%
  filter(!is_practice) %>%
  mutate(
    nudge = trial_nudge == 'default',
    chose_nudge = as.integer(chose_nudge),
    test_trial = trial_num - 2,
    total_points = net_earnings * 3000,
    nudge = ifelse(nudge,'Present','Absent')
  ) %>%
  group_by(test_trial,nudge) %>%
  summarize(average_earnings = mean(total_points)) %>%
  ggplot(aes(x=test_trial,y=average_earnings,color=nudge)) +
  geom_smooth(alpha=0.2) +
  stat_summary_bin(fun.data=mean_se, bins=5) +
  nudge_colors +
  scale_x_continuous(limits = c(0,32)) +
  labs(x="Trial Number", y="Net Earnings")

p5 = p3+p4

savefig("default_learning_curves", 7, 3)
  

# %% ==================== Stats ====================

# move me to base.r
error_table = function(df, yvar, ...) {
    df %>%
        group_by(model, ...) %>%
        summarise(y=mean({{yvar}})) %>%
        pivot_wider(names_from=model, values_from=y) %>%
        mutate(error = Model - Human)
}

df %>%
    error_table(chose_nudge, nudge, cond)

human2 = human %>% mutate(
    nudge = as.integer(nudge),
    many_options = as.integer(n_option == 5),
    many_features = as.integer(n_feature == 5),
)
# H1: Relative probability of selecting the default is positive

glm(chose_nudge ~ nudge, data = human2, family='binomial') %>% 
    write_model("{path}/choice_simple")

human2 %>% 
    group_by(nudge) %>% 
    summarise(prop=100*mean(chose_nudge)) %>% 
    rowwise() %>% group_walk(~ with(.x, 
        write_tex("{path}/choice_percentage/nudge{nudge}", "{prop:.1}")
    ))

# H2: The relative probability of selecting the default will be higher wheen there are many options
# H3: The relative probability of selecting the default will be higher when there are many features
# H4: The relative probability of selecting the default will be lower on trials with higher weights deviation 
glm(chose_nudge ~ nudge * (many_options + many_features + weight_dev),data=human2,family='binomial') %>% 
    write_model("{path}/choice_interactions")

# H5: Metalevel reward will be higher when the default is shown
lm(total_points ~ nudge, data=human2) %>% 
    write_model("{path}/metalevel_return")

human2 %>% 
    group_by(nudge) %>% 
    summarise(total=mean(total_points)) %>% 
    rowwise() %>% group_walk(~ with(.x, 
        write_tex("{path}/metalevel_return_amounts/nudge{nudge}", "{total:.2}")
    ))

# H6: Trials with higher weight deviation will benefit have less benefit from default nudges
lm(total_points ~ nudge * weight_dev, data=human2) %>% 
    write_model("{path}/metalevel_return_interaction")

# H7: The relative probability of selecting the default will be positive on trials where participants revealed at least one value
human2 %>% 
    filter(num_values_revealed > 0) %>% 
    glm(chose_nudge ~ nudge, data=., family='binomial') %>% 
    write_model("{path}/choice_revealed")

human2 %>% 
    filter(num_values_revealed > 0) %>% 
    group_by(nudge) %>% 
    summarise(prop=100*mean(chose_nudge)) %>% 
    rowwise() %>% group_walk(~ with(.x, 
        write_tex(glue("{path}/choice_percentage_revealed/nudge{nudge}"), "{prop:.1}")
    ))

df %>%
    filter(num_values_revealed > 0) %>% 
    group_by(model, nudge) %>% 
    summarise(prop=100*mean(chose_nudge)) %>% 
    rowwise() %>% group_walk(~ with(.x, 
        write_tex(glue("{path}/choice_percentage_revealed/{model}_{nudge}"), "{prop:.1}")
    ))

df %>% 
    group_by(model, nudge) %>% 
    summarise(prop=100*mean(num_values_revealed == 0)) %>% 
    rowwise() %>% group_walk(~ with(.x, {
        write_tex("{path}/no_deliberation/{model}_{nudge}", "{prop:.1}")
        write_tex(glue("{path}/no_deliberation/{model}_{nudge}_inv"), "{100-prop:.1}")
    }))



# %% ==================== Explore ====================
quit()  # don't run below in script
# %% --------
df %>% 
    mutate(no_click = num_values_revealed == 0) %>% 
    group_by(model, no_click, nudge) %>% 
    summarise(mean(chose_nudge))
# %% --------
df %>% 
    filter(num_values_revealed > 0) %>% 
    group_by(model, nudge) %>% 
    summarise(mean(chose_nudge))

# %% --------

df %>% 
    mutate(no_reveal = num_values_revealed == 0) %>% 
    ggplot(aes(nudge, chose_nudge, color=no_reveal, group=no_reveal)) +
        stat_summary(fun=mean, geom="line") +
        point_and_error + 
        facet_rep_grid(~model) + 
        labs(linetype="Options", x='Nudge', y='Prob Choose Default')

# %% --------



p1 = df %>% 
    ggplot(aes(nudge, num_values_revealed, color=n_feature, group=cond)) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        point_and_error + 
        feature_colors +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid"))

savefig("tmp", 7, 3)
# %% --------

p3 = df %>%
    ggplot(aes(weight_dev, chose_nudge, color=nudge, fill=nudge)) + 
    geom_smooth(alpha=0.2) +
    stat_summary_bin(fun.data=mean_se, bins=5) +
    facet_rep_wrap(~model) +
    xlim(0, 40) +
    nudge_colors +
    labs(x="Preference Idiosyncrasy", y="Prob Choose Default")

fig("tmp", 7, 3)

# %% --------
p1 = df %>%
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
    #point_and_error + 
    # nudge_colors +
    # facet_rep_wrap(~model)
    facet_rep_grid(interaction(n_feature, n_option) ~ model, scales="free_y")

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
        #point_and_error + 
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
        #point_and_error + 
        nudge_colors +
        facet_rep_grid(~model) + 
        scale_linetype_manual(values=c("dotted", "solid"))

savefig("tmp", 7, 3)

# %% --------
df %>% 
    ggplot(aes(n_option, chose_nudge, color=nudge, group=interaction(nudge,n_feature))) +
        stat_summary(aes(linetype=n_feature), fun=mean, geom="line") +
        #point_and_error + 
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



















p1 = df %>% 
  ggplot(aes(nudge, chose_nudge)) +
  stat_summary(fun=mean, geom="point") +
  # point_and_error + 
  stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=.15, size=.5) + 
  feature_colors +
  facet_rep_grid(~model) + 
  scale_linetype_manual(values=c("dotted", "solid")) +
  labs(linetype="Options", x='Nudge', y='Prob Choose Default')
savefig("overall-means", 7, 3)




p1 = df %>% 
  ggplot(aes(nudge, chose_nudge)) +
  stat_summary(fun=mean, geom="point") +
  # point_and_error + 
  stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=.2, size=.5) + 
  feature_colors +
  facet_rep_grid(~model) + 
  scale_linetype_manual(values=c("dotted", "solid")) +
  labs(linetype="Options", x='Nudge', y='Prob Choose Default')
savefig("overall-means", 7, 3)

p2 = df %>% 
  subset(num_values_revealed != 0) %>%
  ggplot(aes(nudge, chose_nudge)) +
  stat_summary(fun=mean, geom="point") +
  # point_and_error + 
  stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=.2, size=.5) + 
  feature_colors +
  facet_rep_grid(~model) + 
  scale_linetype_manual(values=c("dotted", "solid")) +
  labs(linetype="Options", x='Nudge', y='Prob Choose Default')


savefig("effects_nudge_payoff_big_facet", 7, 3)


p1 = df %>% 
  ggplot(aes(nudge, total_points)) +
  stat_summary(fun=mean, geom="point") +
  # point_and_error + 
  stat_summary(fun.data=mean_cl_boot, geom="errorbar", width=.2, size=.5) + 
  feature_colors +
  facet_rep_grid(~model) + 
  scale_linetype_manual(values=c("dotted", "solid")) +
  labs(linetype="Options", x='Nudge', y='Net earnings')
savefig("nudge-effects-earnings", 7, 3)




df %>%
  ggplot(aes(weight_dev, total_points, color=nudge, fill=nudge)) + 
  geom_smooth(alpha=0.2) +
  #stat_summary_bin(fun.data=mean_se, bins=5) +
  facet_rep_wrap(~model) +
  xlim(0, 40) +
  nudge_colors +
  payoff_line_lims +
  labs(x="Preference Idiosyncrasy", y="Net Earnings")


