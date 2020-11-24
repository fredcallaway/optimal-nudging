df = load_sim('suggest_new_sims')

# %% ==================== Summary stats ====================

choose = df.groupby(['after', 'reveal_cost', 'n_option', 'n_feature']).choose_suggested.mean()
choose_gain = choose[0] - choose[1]

# %% ==================== Suggest early ====================


pal = {0: 'C2', 1: '#AED1B2'}
g = catplot(df, 'reveal_cost', 'choose_suggested', hue='after',
        palette=pal)
g.set(ylim=(0, 1.05), yticks=(0, 1))

g.axes[1,0].annotate('early suggestion', (.05, 0.27), xycoords='axes fraction', 
    color=pal[0], horizontalalignment='left', fontweight='bold')
g.axes[1,0].annotate('late suggestion', (.05, 0.08), xycoords='axes fraction', 
    color=pal[1], horizontalalignment='left', fontweight='bold')

show('suggest_early')

# %% ==================== Suggest new deviation ====================

g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

def plot_one(data, **kwargs):
    data['x'] = pd.qcut(data.weight_dev, 4).apply(lambda x: x.mid)
    data.groupby('x').choose_suggested.mean().plot()

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xticklabels([1,2,3,4])
g.set_xlabels('Weight Deviation Quartile')
g.set_ylabels('Prob Choose Suggested')
show()

# %% ==================== Payoff ====================

g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=3)

def plot_one(data, **kwargs):
    for nudge, d in data.groupby('nudge'):
        c = PAL[nudge]
        g = d.groupby('cost')
        payoff = g.payoff.mean()
        meta = g.meta_return.mean()
        payoff.plot(color=c)
        plt.fill_between(meta.index, meta, payoff, alpha=0.1, color=c)
        meta.plot(color=c, lw=3)
    plt.xticks(())

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Cost')
g.set_ylabels('Utility')
show('suggest_utility', pdf=True)
