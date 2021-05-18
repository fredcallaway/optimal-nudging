source("base.r")


# %% ==================== Load data ====================

m = read_csv('../model/results/attention_sims.csv') %>% 
    # filter(reveal_cost == only(unique(h$reveal_cost))) %>% 
    # select(-n_click_default) %>% 
    mutate(participant_id = "model")

df = m %>% mutate(
    nudge = factor(nudge, levels=c(0,1), labels=c("absent", "present")),
    model = participant_id == "model"
)

# %% ==================== Plot utils ====================


plot_both = function(yvar) {
    df %>% 
        filter(reveal_cost == 3) %>% 
        ggplot(aes(weight_highlight, {{yvar}}, color=nudge)) +
        stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
        # geom_smooth() + 
        facet_wrap(~log(α), labeller = label_glue("log(α) = {`log(α)`}")) +
        theme(
            legend.position="top", 
            axis.line = element_blank(),
            panel.border = element_rect(colour = "black", fill = NA, size=1)
        )
    fig()
}

plot_difference = function(yvar, ylab) {
    X = df %>% 
        group_by(α, reveal_cost, n_option, n_feature, weight_highlight, nudge) %>% 
        summarise(yvar = mean({{yvar}})) %>% 
        pivot_wider(names_from=nudge, values_from=yvar) %>% 
        mutate(nudge_effect = present - absent)

    X %>% 
        ggplot(aes(weight_highlight, nudge_effect)) +
        stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
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
plot_difference(n_click_highlight, "increase in value of highlighted feature")

plot_both(value_highlight)
plot_difference(value_highlight, "increase in value of highlighted feature")

plot_both(as.numeric(highlight_loss==0))
plot_difference(as.numeric(highlight_loss==0), "increase in probability of\nmaximizing highlighted feature")


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