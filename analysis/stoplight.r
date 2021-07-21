source("base.r")

h = read_csv('../data/final_experiments_data/stoplight_data.csv') %>%
    filter(!is_practice) %>% 
    transmute(
        model="Human",
        participant_id = participant_id,
        n_option = num_options,
        n_feature = num_features,
        decision_cost = points_click_cost,
        reveal_cost = original_item_cost,
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = points_action_utility,
        n_click_highlight = highlight_num_reveals,
        highlight_loss = highlight_value - max_highlight_value,
        weight_highlight = highlight_weight,
        highlight_value,
        nudge = factor(!is_control, levels=c(F,T), labels=c("absent", "present"))
    )

m = read_csv('../model/results/attention_sims.csv') %>% 
    filter(reveal_cost == 3) %>% 
    mutate(
        model="Model",
        participant_id = -1,
        nudge = factor(nudge, levels=c(0,1), labels=c("absent", "present")),
    )

df = bind_rows(h, m)

# %% --------

plot_both = function(yvar) {
    df %>% 
        # group_by(agent) %>% 
        # slice_sample(n=500) %>% 
        ggplot(aes(weight_highlight, {{yvar}}, color=nudge, fill=nudge)) +
        geom_smooth(alpha=0.2) +
        stat_summary_bin(fun.data=mean_se, bins=5) +
        facet_rep_grid(~model) +
        theme(
            # legend.position="top", 
            # axis.line = element_blank(),
            # panel.border = element_rect(colour = "black", fill = NA, size=1)
        ) + scale_colour_manual(values=c(
            "darkgray",
            "dodgerblue"
        ), aesthetics=c("fill", "colour"), name="Nudge")
}

p1 = plot_both(n_click_highlight) + 
    labs(y="Highlight Reveals") +
    theme(axis.title.x=element_blank())#, axis.text.x=element_blank())

p2 = plot_both(highlight_value) +
    labs(x="Weight of Highlighted Feature", y="Higlight Value")
# + labs(x="", y="Number of Reveals\nfor Highlighted Feature")
# + labs(x="Weight of Highlighted Feature", y="Highlighted Feature Value\nof Chosen Option")



(p1 / p2) + plot_layout(guides = "collect")
fig("stoplight", 7, 6)



