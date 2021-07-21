source("base.r")


# %% ==================== Load data ====================

h = read_csv('../data/final_experiments_data/default_data.csv') %>%
    filter(!is_practice) %>% 
    transmute(
        participant_id = as.character(participant_id),
        n_option = as.factor(og_baskets),
        n_feature = as.factor(num_features),
        reveal_cost = cost,
        nudge = trial_nudge == 'default',
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = gross_earnings,        
        choose_default = as.integer(chose_nudge),
    )

m = read_csv('../model/results/default_sims.csv') %>% 
    filter(reveal_cost == only(unique(h$reveal_cost))) %>% 
    select(-n_click_default) %>% 
    mutate(
        participant_id = "model",
        n_option = as.factor(n_option),
        n_feature = as.factor(n_feature)
    )

df = bind_rows(m, h) %>% mutate(
    nudge = factor(nudge, levels=c(0,1), labels=c("absent", "present")),
    model = if_else(participant_id == "model", "Model", "Human"),
    cond = glue("{n_option} Options {n_feature} Features")
)

# %% ==================== Plot utils ====================

option_feature_grid = facet_grid(n_option ~ n_feature, 
    labeller = label_glue(
        rows = "{n_option} Options",
        cols = "{n_feature} Features"
    )
)

option_feature_grid_rep = facet_rep_grid(n_option ~ n_feature, 
    labeller = label_glue(
        rows = "{n_option} Options",
        cols = "{n_feature} Features"
    )
)

chance_line = geom_hline(aes(yintercept = 1/n_option), lty="dotted")

# %% ==================== One panel ====================

df %>% mutate(cond = glue("{n_option} Options {n_feature} Features")) %>% 
    ggplot(aes(nudge, choose_default, color=cond, group=cond)) +
        stat_summary(fun.data=mean_se, size=.2) +
        stat_summary(fun.data=mean_se, geom="line") +
        scale_colour_manual(values=c(
            "gray",
            "dodgerblue",
            "red2",
            "purple2"
        ), aesthetics=c("fill", "colour"), name="") +
    facet_wrap(~model)

fig(w=7, h=3)

# %% --------

df %>% 
    ggplot(aes(nudge, choose_default, color=n_feature, group=cond)) +
        stat_summary(fun.data=mean_se, geom="pointrange", size=.2) +
        stat_summary(aes(linetype=n_option), fun=mean, geom="line") +
        facet_wrap(~model) + 
        scale_colour_manual(values=c(
            "gray", "red2"
        ), aesthetics=c("fill", "colour"), name="Features") +
        labs(linetype="Options")

fig(w=7, h=3)

# %% ==================== Lines ====================

ggplot(df, aes(nudge, choose_default, color=model, group=model, linetype=model)) +
    stat_summary(geom="line", fun.data=mean_se) +
    stat_summary(geom="linerange", fun.data=mean_cl_normal, size=0.8, data=filter(df, !model)) +
    # stat_summary(geom="pointrange", fun.data=mean_cl_normal, fatten=0, size=1) +
    # stat_summary(geom="errorbar", width=0.2, fun.data=mean_se, data=filter(df, !model)) +
    option_feature_grid_rep +
    chance_line +
    # coord_capped_cart(left='both') +
    scale_y_continuous("Probability Choose Default", breaks=c(0,1), limits=c(0,1)) +
    scale_x_discrete(NULL, labels=c("Without\nNudge", "With\nNudge")) +
    scale_colour_manual(values=c("#111111", RED), aesthetics="color") +
    theme(
        legend.position="none", 
        # axis.line = element_blank(),
        panel.border = element_blank()
    )

fig("lines")

# %% ==================== Boxed lines ====================

ggplot(df, aes(nudge, choose_default, color=model, group=model, linetype=model)) +
    stat_summary(geom="line", fun.data=mean_se) +
    stat_summary(geom="linerange", fun.data=mean_cl_normal, size=0.8, data=filter(df, !model)) +
    # stat_summary(geom="pointrange", fun.data=mean_cl_normal, fatten=0, size=1) +
    # stat_summary(geom="errorbar", width=0.2, fun.data=mean_se, data=filter(df, !model)) +
    option_feature_grid + 
    chance_line +
    # theme(panel.grid.major.y = element_line(color="#E9E9E9")) +
    scale_y_continuous("Probability Choose Default", breaks=c(0,1), limits=c(0,1)) +
    scale_x_discrete(NULL, labels=c("Without\nNudge", "With\nNudge")) +
    scale_colour_manual(values=c("#111111", RED), aesthetics="color") +
    theme(
        legend.position="none", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )

fig("boxes")

# %% ==================== Bar + Points ====================

ggplot(df, aes(nudge, choose_default, group=model)) +
    stat_summary(geom="bar", fun=mean, data=filter(df, !model),  fill="white", color=BLACK) +
    stat_summary(geom="errorbar", width=0.2, fun.data=mean_cl_normal, data=filter(df, !model)) +
    stat_summary(geom="point", fun=mean, data=filter(df, model), color=RED, shape="diamond", size=2) +
    option_feature_grid_rep + 
    theme(axis.ticks.x = element_blank()) + 
    geom_hline(aes(yintercept = 1/n_option), lty="dashed") +
    scale_x_discrete(NULL, labels=c("Without\nNudge", "With\nNudge")) +
    scale_y_continuous("Probability Choose Default", breaks=c(0,1), limits=c(0,1)) +
    theme(legend.position="none") + 

fig("bars")

# %% ==================== Bars (no model) ====================


# facet_wrap(~ n_option + n_feature,
#     labeller = label_glue("{n_option} Options - {n_feature} Features")
# ) + 

ggplot(filter(df, !model), aes(nudge, choose_default, fill=nudge)) +
    stat_summary(geom="bar", fun=mean) +
    stat_summary(geom="errorbar", width=0.2, fun.data=mean_se) +
    option_feature_grid +
    chance_line +
    scale_y_continuous("Probability Choose Default", breaks=c(0,1), limits=c(0,1)) +
    scale_colour_manual(values=c(GRAY, BLUE), aesthetics="fill") +
    theme(legend.position="none")

fig("bars")

# %% --------