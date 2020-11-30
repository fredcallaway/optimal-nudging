def load_data(name):
    name = 'supersize_data'
    df = pd.read_csv(f'../data/{name}.csv')
    df.rename(columns={
        'cost': 'reveal_cost', 
        'chose_nudge': 'choose_suggested',
        'og_baskets': 'n_option',
        'num_features': 'n_feature',
        'weights_deviation': 'weight_dev',
        'net_earnings': 'meta_return'

    }, inplace=True)
    df = df.query('trial_nudge != "control"').copy()
    df['after'] = (df.trial_nudge == 'post-supersize').astype(int)
    return df

data = {
    'model': load_sim('suggest_new_sims', cost=2),
    'human': load_data('supersize_data')
}

# %% ==================== Main figure ====================

def suggestion(df, agent):
    g = sns.catplot('n_feature', 'choose_suggested',
        data=df,
        hue='after', col='naive', height=4,
        palette={0: 'C2', 1: '#AED1B2'},
        kind='point', legend=False)

    for ax, t in zip(g.axes.flat, ['Savvy', 'Naïve']):
        ax.set_title(t)

    g.set_xlabels("")
    g.set_xticklabels(["2 Features", "5 Features"])
    g.set_ylabels('Prob Choose Suggested')
    plot_chance(g, [1/6])

    g.set(ylim=(0, 1), yticks=(0, 1))

plot_both(suggestion, data)

# %% --------
g = catplot(data['model'].query('reveal_cost == 2'),
   'naive', 'choose_suggested', col='n_feature',
    hue='after', palette={0: 'C2', 1: '#AED1B2'},
    kind='point', legend=False)
show()

# %% --------
g = catplot(data['model'].query('reveal_cost == 4'),
   'naive', 'choose_suggested', 
    hue='after', palette={0: 'C2', 1: '#AED1B2'},
    kind='point', legend=False)


ax = g.axes.flat[0]
handles, labels = ax.get_legend_handles_labels()
new_labels = ['Early', 'Late']
ax.legend(handles=handles, labels=new_labels)
plot_chance(g, [1/3, 1/6])
g.set(ylim=(0, 1.05), yticks=(0, 1))
g.set_xlabels("")
g.set_xticklabels(['Savvy', 'Naïve'])


show()

# %% ==================== Benefits ====================
def suggestion_benefits(df, agent):
    g = sns.catplot('naive', 'meta_return',
        data=df,
        hue='after', col='n_feature', height=4,
        palette={0: 'C2', 1: '#AED1B2'},
        sharey=False,
        kind='point', legend=False)

plot_both(suggestion_benefits, data)


# %% ==================== After x Naive ====================
df = data['model'].query('n_feature == 5 and n_option == 5')

g = sns.catplot('reveal_cost', 'choose_suggested', data=df, 
    hue='after', col='naive', palette={0: 'C2', 1: '#AED1B2'},
    kind='point', legend=False)
ax = g.axes.flat[0]
handles, labels = ax.get_legend_handles_labels()
new_labels = ['Early', 'Late']
ax.legend(handles=handles, labels=new_labels)
for ax, t in zip(g.axes.flat, ['Savvy', 'Naïve']):
    ax.set_title(t)

g.set_xlabels('Reveal Cost')
g.set_ylabels('Prob Choose Suggested')

show()

# %% --------



# %% ==================== Suggest early ====================

def suggest_early(df, agent):
    pal = {0: 'C2', 1: '#AED1B2'}
    g = catplot(df, 'reveal_cost', 'choose_suggested', hue='after',
            palette=pal, kind='point')
    g.set(ylim=(0, 1.05), yticks=(0, 1))

    if agent == 'model':
        g.axes[1,0].annotate('early suggestion', (.05, 0.27), xycoords='axes fraction', 
            color=pal[0], horizontalalignment='left', fontweight='bold')
        g.axes[1,0].annotate('late suggestion', (.05, 0.08), xycoords='axes fraction', 
            color=pal[1], horizontalalignment='left', fontweight='bold')


# plot_both(suggest_early, data)
df = data['model'].query('naive == 1')
suggest_early(df, 'model')
show()

# %% ==================== Suggest new deviation ====================
import json
def mad(x):
    return np.mean(np.abs(x - np.mean(x)))

df.weights = df.weights.apply(json.loads)
df.weight_dev = df.weights.apply(mad)
# %% --------

def suggest_deviation(df, agent):
    g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

    def plot_one(data, **kwargs):
        data['x'] = pd.qcut(data.weight_dev, 4).apply(lambda x: x.mid)
        data.groupby('x').choose_suggested.mean().plot()

    g.map_dataframe(plot_one)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_xticklabels([1,2,3,4])
    g.set_xlabels('Weight Deviation Quartile')
    g.set_ylabels('Prob Choose Suggested')

plot_both(suggest_deviation, data)

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
