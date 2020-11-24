data = {
    'model': load_sim('default_sims'),
    'human': load_sim('default_sims')
}

# %% ==================== Summary stats ====================

payoff = df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).payoff.mean()
payoff_gain = payoff[1] - payoff[0]

cost = df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).decision_cost.mean()
cost_gain = cost[0] - cost[1]


df['nocost'] = df.decision_cost == 0
df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).nocost.mean()[1]


# %% ==================== Prob choose default ====================

def choose_default(df):
    g = catplot(df, 'reveal_cost', 'choose_default', hue='nudge', palette=PAL)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_xlabels('Reveal Cost')
    g.set(ylim=(0, 1.05), yticks=(0, 1))
    g.set_ylabels('Prob Choose Default')

    g.axes[1,0].annotate('with default', (.05, 0.9), xycoords='axes fraction', 
        color=PAL[1], horizontalalignment='left', fontweight='bold')
    g.axes[1,0].annotate('without default', (.05, 0.48), xycoords='axes fraction', 
        color=PAL[0], horizontalalignment='left', fontweight='bold')
    return g

plot_both(choose_default, data)

# %% ==================== Default benefits  ====================

def default_utility(df):
    g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

    def plot_one(data, **kwargs):
        n = 4
        data['wvb'] = pd.cut(data.weight_dev, n)
        for nudge, d in data.groupby('nudge'):
            c = PAL[nudge]
            g = d.groupby('wvb')
            payoff = g.payoff.mean()
            meta = g.meta_return.mean()
            payoff.plot(color=c)
            plt.fill_between(range(0,n), meta, payoff, alpha=0.1, color=c)
            meta.plot(color=c, lw=3)
        # plt.yticks([], [])

    g.map_dataframe(plot_one)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_ylabels('Utility')
    g.set_xticklabels([1,2,3,4])
    g.set_xlabels('Weight Deviation Quartile')

    g.axes[1,0].annotate('action utility', (.84, 0.83), xycoords='axes fraction', 
        color=PAL[0], horizontalalignment='right', rotation=17)

    g.axes[1,0].annotate('total utility', (.4, 0.4), xycoords='axes fraction', 
        color=PAL[0], horizontalalignment='right', fontweight='bold', rotation=13)

    g.axes[1,0].annotate('deliberation cost', (.55, 0.55), xycoords='axes fraction', 
        color=PAL[0], horizontalalignment='right', rotation=13, alpha=0.65)

plot_both(default_utility, data)


# %% ==================== Weight D  ====================

g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

def plot_one(data, **kwargs):
    data['x'] = pd.qcut(data.weight_dev, 4).apply(lambda x: x.mid)
    for nudge, d in data.groupby('nudge'):
        c = PAL[nudge]
        g = d.groupby('x')
        g.choose_default.mean().plot(color=c)

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')

g.set_xticklabels([1,2,3,4])
g.set_xlabels('Weight Deviation Quartile')
show()
