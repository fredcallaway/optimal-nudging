import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib as mpl
import itertools as it
import os

sns.set_style('ticks')
mpl.rc('lines', linewidth=2.0)
sns.set_context('notebook', font_scale=1.5)

from figures import Figures
figs = Figures('figs', pdf=True)
figs.watch()
show = figs.show; figure = figs.figure

GRAY = (0.7, 0.7, 0.7)
PAL = {0: GRAY, 1: 'C0'}

def load_sim(name):
    df = pd.read_csv(f'../model/results/{name}.csv')
    df['meta_return'] = df.payoff - df.decision_cost
    df.reveal_cost = df.reveal_cost.astype(int)

    return df

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

# %% ==================== Suggest New ====================

df = load_sim('suggest_new_sims')

df['meta_return'] = df.payoff - df.decision_cost

g = sns.catplot('reveal_cost', 'choose_suggested', 
    row='n_option', col='n_feature', data=df, 
    legend=False, kind='point', height=4)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Cost')
g.set_ylabels('Prob Choose Suggested')
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


