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

# %% ==================== Summary ====================

df %>% ggplot(aes(weight_highlight)) + geom_bar()
fig()



# %% ==================== Effect on clicks ====================

df %>% 
    filter(reveal_cost == 3) %>% 
    ggplot(aes(weight_highlight, n_click_highlight, color=nudge)) +
    stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
    # geom_smooth() + 
    option_feature_grid + 
    theme(
        legend.position="top", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )
fig()

# %% --------

X = df %>% 
    group_by(reveal_cost, n_option, n_feature, weight_highlight, nudge) %>% 
    summarise(n_click_highlight = mean(n_click_highlight)) %>% 
    pivot_wider(names_from=nudge, values_from=n_click_highlight) %>% 
    mutate(nudge_effect = present - absent)

X %>% 
    ggplot(aes(weight_highlight, nudge_effect, color=factor(reveal_cost))) +
    stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
    # geom_smooth() + 
    option_feature_grid + 
    labs(y="increase in CLICKS on\nhighlighted feature") + 
    theme(
        legend.position="top", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )
fig()



# %% ==================== Effect on highlighted value ====================



df %>% 
    filter(reveal_cost == 3) %>% 
    ggplot(aes(weight_highlight, highlight_value, color=nudge)) +
    stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
    # geom_smooth() + 
    option_feature_grid + 
    theme(
        legend.position="top", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )
fig()
# %% --------

X = df %>% 
    group_by(reveal_cost, n_option, n_feature, weight_highlight, nudge) %>% 
    summarise(highlight_value = mean(highlight_value)) %>% 
    pivot_wider(names_from=nudge, values_from=highlight_value) %>% 
    mutate(nudge_effect = present - absent)

X %>% 
    ggplot(aes(weight_highlight, nudge_effect, color=factor(reveal_cost))) +
    stat_summary_bin(fun.data=mean_se, bins=20, geom="line") +
    # geom_smooth() + 
    option_feature_grid + 
    labs(y="increase in VALUE of\nhighlighted feature") + 
    theme(
        legend.position="top", 
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, size=1)
    )
fig()