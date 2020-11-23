

# %% ==================== Choose suggestion ====================

df = load_sim('suggest_sims')
df['meta_return'] = df.payoff - df.decision_cost

g = sns.catplot('reveal_cost', 'choose_suggested', 
    row='n_option', col='n_feature', hue='nudge', data=df, 
    legend=False, palette=PAL, kind='point', height=4)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Cost')
g.set_ylabels('Prob Choose Suggested')
show('choose_suggested')

# %% ====================  ====================

df = load_sim('suggest_sims')
df['meta_return'] = df.payoff - df.decision_cost
df['rv_mean'] = df.total_val - df.mean_other
df['rv_max'] = df.total_val - df.max_other

g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

def plot_one(data, **kwargs):
    data['x'] = pd.qcut(data.weight_dev, 4).apply(lambda x: x.mid)
    for nudge, d in data.groupby('nudge'):
        c = PAL[nudge]
        g = d.groupby('x')
        g.choose_suggested.mean().plot(color=c)

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')

g.set_xticklabels([1,2,3,4])
g.set_xlabels('Weight Deviation Quartile')

# g.set(xlim=(1, 10), xticks=[])
# g.set_xticklabels(())
# g.set_xlabels('Something')
# g.set_xlabels('Average Non-Best Feature Value')
g.set_ylabels('Prob Choose Suggested')
show()

# %% --------

data = df.query('n_option == 5 and n_feature == 2')
data['x'] = pd.cut(data.other_val, 10).apply(lambda x: x.mid)
sns.pointplot('x', 'choose_suggested', data=data, hue='nudge')
show()
