source("base.r")
path = paste0("stats/stoplight", if (EXCLUDE) "" else "-full")
require(gridExtra)

# %% ==================== Load data ====================

human_raw = read_csv('../data/experiments/reported_experiments/stoplight_data.csv', col_types = cols())
model_raw = read_csv('../model/results/stoplight_sims.csv', col_types = cols())

# %% --------
human = human_raw %>%
    filter(!is_practice) %>% 
    transmute(
        highlight_value, 
        num_values_revealed,
        maximized_highlighted_option,
        participant_id = as.character(participant_id),
        n_option = num_options,
        n_feature = num_features,
        decision_cost = points_click_cost,
        reveal_cost = original_item_cost,
        weight_dev = weights_deviation,
        decision_cost = click_cost,
        payoff = points_action_utility,
        n_click_highlight = highlight_num_reveals,
        # highlight_loss = highlight_value - max_highlight_value,
        weight_highlight = highlight_weight,
        nudge = factor(!is_control, levels=c(F,T), labels=c("Absent", "Present"))
    ) %>% apply_exclusion(nudge == "Absent")

report_exclusion(path, human_raw, human)

model = model_raw %>% 
    filter(reveal_cost == 3) %>% 
    mutate(
        participant_id="model",
        nudge = factor(nudge, levels=c(0,1), labels=c("Absent", "Present")),
    )

df = bind_rows(human, model) %>% mutate(
    model = factor(if_else(participant_id == "model", "Model", "Human"), levels=c("Model", "Human")),
)

# %% ==================== Plot ====================

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
        ) + nudge_colors
}

p1 = plot_both(n_click_highlight) + 
    labs(y="Highlight Reveals") +
    theme(axis.title.x=element_blank())#, axis.text.x=element_blank())

p2 = plot_both(highlight_value) +
    labs(x="Weight of Highlighted Feature", y="Higlight Value")
# + labs(x="", y="Number of Reveals\nfor Highlighted Feature")
# + labs(x="Weight of Highlighted Feature", y="Highlighted Feature Value\nof Chosen Option")

(p1 / p2) + plot_layout(guides = "collect") + plot_annotation(tag_levels = 'A')

savefig("stoplight", 7, 6)


## learning curves: no exclusion
save_stoplight_learning_curves = function(exclusion){
  
  p3 = human_raw %>%
    filter(!is_practice) %>%
    mutate(
      nudge  = ifelse(is_control,'Absent','Present'),
      test_trial = trial_num - 2
    ) %>%
    {if (exclusion) apply_exclusion(.,nudge == 'Absent') else .} %>%
    group_by(test_trial,nudge) %>%
    summarize(average_n_highlight = mean(highlight_num_reveals)) %>%
    ggplot(aes(x=test_trial,y=average_n_highlight,color=nudge)) +
    geom_smooth(alpha=0.2) +
    stat_summary_bin(fun.data=mean_se, bins=5) +
    nudge_colors +
    scale_x_continuous(limits=c(0,31)) +
    labs(x="Trial Number", y="Highlight Reveals") +
    theme(legend.position = 'none')
  
  p4 = human_raw %>%
    filter(!is_practice) %>%
    mutate(
      nudge  = ifelse(is_control,'Absent','Present'),
      test_trial = trial_num - 2
    ) %>%
    {if (exclusion) apply_exclusion(.,nudge == 'Absent') else .} %>%
    group_by(test_trial,nudge) %>%
    summarize(average_highlight_value = mean(highlight_value)) %>%
    ggplot(aes(x=test_trial,y=average_highlight_value,color=nudge)) +
    geom_smooth(alpha=0.2) +
    stat_summary_bin(fun.data=mean_se, bins=5) +
    nudge_colors +
    scale_x_continuous(limits=c(0,31)) +
    labs(x="Trial Number", y="Highlight Value")
  
  return(p3 + p4)
}

p5 = save_stoplight_learning_curves(F)
p6 = save_stoplight_learning_curves(T)
(p5 / p6) + plot_annotation(tag_levels = list(c('A', '','B','')))
savefig("stoplight_learning_curves", 7, 6)

human_raw %>%
  filter(!is_practice) %>%
  mutate(
    Nudge  = ifelse(is_control,'Absent','Present'),
    test_trial = trial_num - 2
  ) %>%
  group_by(test_trial,Nudge) %>%
  summarize(average_highlight_value = mean(highlight_value)) %>%
  ggplot(aes(x=test_trial,y=average_highlight_value,color=Nudge)) +
  geom_smooth(alpha=0.2) +
  geom_point() +
  # stat_summary_bin(fun.data=mean_se, bins=5) +
  nudge_colors +
  scale_x_continuous(limits=c(0,31)) +
  labs(x="Trial Number", y="Highlight Value")




# %% ==================== Stats ====================

# MSEs
df %>%
  mutate(highlight_bin = sapply(weight_dev,get_highlight_bin)) %>%
  get_squared_error(n_click_highlight, nudge,highlight_bin) %>%
  rowwise() %>% group_walk(~ with(.x, 
    write_tex("{path}/mses/n_highlight_reveals", "{prop:.4}")
  ))

df %>%
  mutate(highlight_bin = sapply(weight_dev,get_highlight_bin)) %>%
  get_squared_error(highlight_value, nudge,highlight_bin) %>%
  rowwise() %>% group_walk(~ with(.x, 
    write_tex("{path}/mses/highlight_value", "{prop:.4}")
  ))

human2 = mutate(human,
    nudge = as.integer(nudge == "Present")
)

# H1: Revealed values higher 
t.test(n_click_highlight ~ nudge, data = human2, alternative='less') %>% 
    tidy %>% 
    with({
        write_tex("{path}/num_reveal/nudge0", "{estimate1:.2}")
        write_tex("{path}/num_reveal/nudge1", "{estimate2:.2}")
        write_tex("{path}/num_reveal/ttest", 
            "$t({parameter:.1}) = {-statistic:.2},\ {pval(p.value)}$"
        )
    })

# H2: Highlight value
t.test(highlight_value ~ nudge, data=human2, alternative='less') %>% 
    tidy %>% 
    with({
        write_tex("{path}/value/nudge0", "{estimate1:.2}")
        write_tex("{path}/value/nudge1", "{estimate2:.2}")
        write_tex("{path}/value/ttest", 
            "$t({parameter:.1}) = {-statistic:.2},\ {pval(p.value)}$"
        )
    })

# H3: Probability of maximizing the highlighted_option
human2 %>% 
    group_by(nudge) %>% 
    summarise(n=n(), n_max=sum(maximized_highlighted_option)) %>% 
    with(prop.test(n_max, n, alternative='less', correct=FALSE)) %>% 
    tidy %>% 
    with({
        write_tex("{path}/maximize/nudge0", "{100*estimate1:.1}")
        write_tex("{path}/maximize/nudge1", "{100*estimate2:.1}")
        write_tex("{path}/maximize/proptest", "$\\chi^2({parameter}) = {round(statistic)},\\ {pval(p.value)}$")
    })

# %% ==================== Explore ====================
quit()  # don't run below in script
# %% --------


