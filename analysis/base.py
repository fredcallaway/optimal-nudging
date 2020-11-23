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

names = {
    'choose_default': 'Prob Choose Default',
    'choose_suggested': 'Prob Choose Suggested',
    'decision_cost': 'Click Cost',
    'reveal_cost': 'Cost per Click',
}

def catplot(data, x, y, **kws):
    g = sns.catplot(x, y, 
        row='n_option', col='n_feature', data=data, 
        legend=False, kind='point', height=4, **kws)
    g.set_titles(template='{row_name} Options – {col_name} Features')
    if x in names:
        g.set_xlabels(names[x])
    if y in names:
        g.set_ylabels(names[y])
    return g