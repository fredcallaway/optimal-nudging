df = load_sim('suggest_new_sims')

catplot(df, 'reveal_cost', 'choose_suggested')
show()

# %% ==================== Suggest new deviation ====================

df = pd.read_csv('results/suggest_new_sims.csv')
df['meta_return'] = df.payoff - df.decision_cost

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
