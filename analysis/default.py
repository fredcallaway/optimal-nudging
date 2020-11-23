
# %% ==================== Prob choose default ====================
df = load_sim('default_sims')

g = sns.catplot('reveal_cost', 'choose_default', 
    row='n_option', col='n_feature', hue='nudge', data=df, 
    legend=False, palette=PAL, ci=None, kind='point', height=4)

g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Reveal Cost')
g.set(ylim=(0, 1.05), yticks=(0, 1))
g.set_ylabels('Prob Choose Default')
show('choose_default', pdf=True)
# df.groupby(['reveal_cost', 'n_option', 'n_feature', 'nudge']).decision_cost.mean()

# %% ==================== Default benefits  ====================

g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

def plot_one(data, **kwargs):
    data['wvb'] = pd.cut(data.weight_dev, 4)
    for nudge, d in data.groupby('nudge'):
        c = PAL[nudge]
        g = d.groupby('wvb')
        payoff = g.payoff.mean()
        meta = g.meta_return.mean()
        payoff.plot(color=c)
        plt.fill_between(range(0,4), meta, payoff, alpha=0.1, color=c)
        meta.plot(color=c, lw=3)
    # plt.yticks([], [])
    

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_ylabels('Utility')
g.set_xticklabels([1,2,3,4])
g.set_xlabels('Weight Deviation Quartile')
# g.set(ylim=(150, 250), yticks=[150, 250], yticklabels=(150, 250))
show('default_utility', pdf=True)



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
