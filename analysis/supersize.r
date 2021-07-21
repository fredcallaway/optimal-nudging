source("base.r")

# %% ==================== Load data ====================

human_raw = read_csv('../data/final_experiments_data/supersize_data.csv') 
model_raw = read_csv('../model/results/supersize_sims.csv')
# %% --------
human = human_raw %>%
    filter(trial_nudge != "control") %>% 
    transmute(
        participant_id = as.character(participant_id),
        n_option = og_baskets,
        n_feature = num_features,
        reveal_cost = cost,
        nudge = factor(trial_nudge, levels=c("pre-supersize", "post-supersize"), labels=c("Early", "Late"), ordered=T),
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = gross_earnings,        
        chose_nudge = as.integer(chose_nudge),
    )
# %% --------
model = model_raw %>% 
    filter(
        reveal_cost == only(unique(human$reveal_cost)) &
        n_option == only(unique(human$n_option)) &
        naive == 1
    ) %>% 
    mutate(
        participant_id = "model", 
        chose_nudge = as.integer(choose_suggested),
        nudge = factor(after, levels=c(0, 1), labels=c("Early", "Late"), ordered=T),
    ) %>% 
    select(-c(naive, after, choose_suggested))

# %% --------
df = bind_rows(human, model) %>% mutate(
    model = if_else(participant_id == "model", "Model", "Human"),
    n_option = as.factor(n_option),
    n_feature = as.factor(n_feature)
)

# %% --------

df %>% 
    ggplot(aes(nudge, chose_nudge, color=n_feature, group=n_feature)) +
    facet_wrap(~model) + 
    stat_summary(fun.data=mean_se, size=.2) +
    stat_summary(fun.data=mean_se, geom="line") +
    scale_colour_manual(values=c(
        "gray", "red2"
    ), aesthetics=c("fill", "colour"), name="Features") +
    coord_cartesian(xlim=c(NULL), ylim=c(0, 0.5))

fig(w=7, h=3)

# %% --------

facets = facet_rep_wrap(~n_feature, 
    labeller = label_glue("{n_feature} Features")
)

chance_line = geom_hline(aes(yintercept = 1/(n_option+1)), lty="dotted")

ggplot(df, aes(nudge, chose_nudge, color=model, group=model, linetype=model)) +
    stat_summary(geom="line", fun.data=mean_se) +
    stat_summary(geom="linerange", fun.data=mean_cl_normal, size=0.8, data=filter(df, !model)) +
    # stat_summary(geom="pointrange", fun.data=mean_cl_normal, fatten=0, size=1) +
    # stat_summary(geom="errorbar", width=0.2, fun.data=mean_se, data=filter(df, !model)) +
    facets +
    chance_line +
    # coord_capped_cart(left='both') +
    scale_y_continuous("Probability\nChoose Suggested", breaks=c(0,1), limits=c(0,1)) +
    # scale_x_discrete(NULL, labels=c("Early\nSuggestion", "Late\nSuggestion")) +
    scale_colour_manual(values=c("#111111", RED), aesthetics="color") +
    theme(
        legend.position="none", 
        axis.title.x=element_blank(),
        # axis.line = element_blank(),
        panel.border = element_blank()
    )

fig("supersize", w=4, h=2.2)
