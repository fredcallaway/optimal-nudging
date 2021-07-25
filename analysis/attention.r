source("base.r")



# highlight_value_check => value of the highlighted option
# num_highlight_reveals => how many highlighted options were revealed
# highlight_weight

# "../data/pilot_data/traffic-light-pilot4.csv"


# %% ==================== Load data ====================

h = read_csv('../data/pilot_data/traffic-light-pilot4.csv') %>%
    filter(!is_practice) %>% 
    transmute(
        agent="human",
        participant_id = participant_id,
        n_option = num_options,
        n_feature = num_features,
        decision_cost = points_click_cost,
        reveal_cost = original_item_cost,
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = points_action_utility,
        n_click_highlight = num_highlight_reveals,
        # highlight_loss = highlight_value_check - h
        weight_highlight = highlight_weight,
        highlight_value = highlight_value_check,
        nudge = factor(!is_control, levels=c(F,T), labels=c("absent", "present"))
    )


m = read_csv('../model/results/attention_sims.csv') %>% 
    mutate(
        agent="model",
        participant_id = -1,
        nudge = factor(nudge, levels=c(0,1), labels=c("absent", "present")),
        model = participant_id == "model"
    )

df = bind_rows(h, m)

# %% ==================== Plot utils ====================

plot_both = function(yvar) {
    df %>% 
        # filter(weight_highlight <= 15) %>% 
        filter(reveal_cost == 3) %>% 
        ggplot(aes(weight_highlight, {{yvar}}, color=nudge)) +
        stat_summary(fun.data=mean_sdl, fun.args=1/sqrt(100)) +
        # geom_smooth() + 
        # facet_wrap(~log(α), labeller = label_glue("log(α) = {`log(α)`}")) +
        facet_wrap(~agent) +
        theme(
            legend.position="top", 
            axis.line = element_blank(),
            panel.border = element_rect(colour = "black", fill = NA, size=1)
        )
    fig()
}
# %% --------

difference_frame = function(df, yvar) {
    df %>%
        filter(weight_highlight <= 15) %>% 
        group_by(α, reveal_cost, n_option, n_feature, weight_highlight, nudge) %>% 
        summarise(yvar = mean({{yvar}})) %>% 
        pivot_wider(names_from=nudge, values_from=yvar) %>% 
        mutate(nudge_effect = present - absent)
}

plot_difference = function(yvar, ylab) {
    difference_frame(df, {{yvar}}) %>%
        ggplot(aes(weight_highlight, nudge_effect)) +
        stat_summary(fun=mean, geom="line") +
        # geom_smooth() + 
        facet_wrap(~log(α), labeller = label_glue("log(α) = {`log(α)`}")) +
        labs(y=ylab) + 
        theme(
            legend.position="top", 
            axis.line = element_blank(),
            panel.border = element_rect(colour = "black", fill = NA, size=1)
        )
    fig()
}


# %% ==================== Plots ====================

df %>% ggplot(aes(weight_highlight)) + geom_bar()
fig()

plot_both(n_click_highlight)
plot_difference(n_click_highlight, "increase in clicks to highlighted feature")

plot_both(highlight_value)
plot_difference(highlight_value, "increase in value of highlighted feature")

plot_both(as.numeric(highlight_loss==0))
plot_difference(as.numeric(highlight_loss==0), "increase in probability of\nmaximizing highlighted feature")

plot_both(decision_cost)
plot_both(payoff - decision_cost)

# %% --------
library(jtools)

df$weight

difference_frame(df, highlight_value) %>% 
    ungroup() %>% 
    mutate(weight_highlight = scale(weight_highlight)) %>%
    lm(nudge_effect ~ weight_highlight + I(weight_highlight^2), data=.) %>% summ

# %% --------

X = df %>% 
    group_by(α, reveal_cost, n_option, n_feature, weight_highlight, nudge) %>% 
    summarise(n_click_highlight = mean(n_click_highlight)) %>% 
    pivot_wider(names_from=nudge, values_from=n_click_highlight) %>% 
    mutate(nudge_effect = present - absent)

X %>% 
    ggplot(aes(weight_highlight, nudge_effect)) +
    stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
    # geom_smooth() + 
    facet_wrap(~log(α), labeller = label_glue("log(α) = {`log(α)`}")) +
    labs(y="increase in CLICKS on\nhighlighted feature") + 
    theme(
        legend.position="top", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )
fig()



# %% ==================== Effect on highlighted value ====================


plot_both(highlight_value)
fig()
# %% --------



fig()