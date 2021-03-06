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

def load_sim(name, cost=None):
    df = pd.read_csv(f'../model/results/{name}.csv')
    df['meta_return'] = df.payoff - df.decision_cost
    if 'reveal_cost' in df:
        df.reveal_cost = df.reveal_cost.astype(int)
    df['choose_immediately'] = df.decision_cost == 0
    if cost is not None:
        return df.query(f'reveal_cost == {cost}')
    else:
        return df

names = {
    'choose_default': 'Prob Choose Default',
    'choose_suggested': 'Prob Choose Suggested',
    'decision_cost': 'Click Cost',
    'reveal_cost': 'Cost per Click',
}

def catplot(data, x, y, **kws):
    kwargs = dict(row='n_option', col='n_feature', data=data, 
                  legend=False, kind='bar', height=4,)
    kwargs.update(kws)
    g = sns.catplot(x, y, **kwargs)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    if x in names:
        g.set_xlabels(names[x])
    if y in names:
        g.set_ylabels(names[y])
    return g

def plot_chance(g, chance):
    for axes, p in zip(g.axes, chance):
        for ax in axes:
            ax.axhline(p, ls=':', color='k')#.set_zorder(0)


def plot_both(func, data):
    for k, v in data.items():
        func(v, k)
        plt.suptitle(k.title(), y=1.03, fontsize=24)
        show(f'{func.__name__}_{k}', pdf=True)
