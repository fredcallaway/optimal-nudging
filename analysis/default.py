
def load_data(name):
    df = pd.read_csv(f'../data/{name}.csv')
    df.rename(columns={
        'cost': 'reveal_cost', 
        'chose_nudge': 'choose_default',
        'og_baskets': 'n_option',
        'num_features': 'n_feature',
        'weights_deviation': 'weight_dev',
        'gross_earnings': 'payoff',
        'net_earnings': 'meta_return'
    }, inplace=True)
    df['choose_immediately'] = df.click_cost <= 1e-7
    df['nudge'] = (df.trial_nudge == 'default').astype(int)
    return df

data = {
    'model': load_sim('default_sims', 1),
    'human': load_data('default_data')
}

# %% ==================== Summary stats ====================

payoff = df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).payoff.mean()
payoff_gain = payoff[1] - payoff[0]

cost = df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).decision_cost.mean()
cost_gain = cost[0] - cost[1]

df['nocost'] = df.decision_cost == 0
df.groupby(['nudge', 'reveal_cost', 'n_option', 'n_feature']).nocost.mean()[1]

# %% --------
delib = df.query(decision_cost > 0)
delib.groupby('nudge').choose_default.mean()

# %% --------
df['click_default'] = df.n_click_default > 0
g = catplot(df.query('decision_cost > 0'), 'nudge', 'click_default', palette=PAL)
show()


# %% ==================== Prob choose default ====================

def choose_default(df, agent):
    g = catplot(df, 'nudge', 'choose_default', palette=PAL)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_xlabels('')
    g.set_xticklabels(['Default\nAbsent', 'Default\nPresent'])
    g.set(ylim=(0, 1.05), yticks=(0, 1))
    g.set_ylabels('Prob Choose Default')
    for i, p in enumerate([0.5, 0.2]):
        for j in [0,1]:
            plt.sca(g.axes[i, j])
            plt.axhline(p, ls=':', color='k')#.set_zorder(0)
    return g

plot_both(choose_default, data)
# choose_default(load_sim('default_sims', 2), 'model')
# show('choose_default_model')

# %% ==================== Not immediately ====================
def not_immediately(df, agent):
    # df = df.query('not choose_immediately')
    g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

    def plot_one(data, **kwargs):
        n = 4
        data['wvb'] = pd.cut(data.weight_dev, n)

        for nudge, d in data.groupby('nudge'):
            c = PAL[nudge]

            d.choose_immediately = pd.Categorical(d.choose_immediately, categories=[False, True])
            g = d.groupby(['choose_immediately', 'wvb'])
            choose_default = g.choose_default.sum()
            P = choose_default / d.groupby('wvb').apply(len)

            x1 = P[True]
            x2 = P[False]

            plt.fill_between(range(len(x1)), x1.values, color=c, alpha=0.3)
            x2 = d.groupby('wvb').choose_default.mean().fillna(0)
            plt.fill_between(range(len(x2)), x2.values, x1.values, color=c, alpha=0.1)
            plt.plot(range(len(x2)), x2.values, lw=2, color=c)

            # d.groupby('wvb').choose_default.mean().astype(float).plot(color=c)
        # plt.yticks([], [])

    g.map_dataframe(plot_one)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_ylabels('Prob Choose Default')
    g.set_xticklabels([1,2,3,4])
    g.set_xlabels('Weight Deviation Quartile')

plot_both(not_immediately, data)
show()

not_immediately(data['human'], 'human')
# df = load_sim('default_sims', 2)
# not_immediately(df, 'model')
# show()
# %% --------

def by_weight(df, agent):
    # df = df.query('not choose_immediately')
    g = sns.FacetGrid(df, 'n_option', 'n_feature', margin_titles=False, height=4)

    def plot_one(data, **kwargs):
        n = 4
        data['wvb'] = pd.cut(data.weight_dev, n)

        for nudge, d in data.groupby('nudge'):
            c = PAL[nudge]
            x = d.groupby('wvb').choose_default.mean()
            plt.plot(range(len(x)), x, color=c, lw=2)

    g.map_dataframe(plot_one)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    g.set_ylabels('Prob Choose Default')
    g.set_xticklabels([1,2,3,4])
    g.set_xlabels('Weight Deviation Quartile')

plot_both(by_weight, data)

# %% --------

d = df.query('n_option == 5 and n_feature == 5 and nudge == 1')
d['wvb'] = pd.cut(d.weight_dev, n)


# %% --------
sns.barplot('nudge', 'choose_default', data=df.query('not choose_immediately'))
show()

# %% ==================== Default benefits  ====================

def default_utility(df, agent='model'):
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
# default_utility(load_sim('default_sims', 3))
# show()

# %% --------
def fooplot(df, agent):
    catplot(df.query('not choose_immediately'), 'nudge', 'choose_default')

plot_both(fooplot, data)

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
