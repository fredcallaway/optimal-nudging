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

from figures import Figures
figs = Figures('figs')
figs.watch()
show = figs.show

GRAY = (0.7, 0.7, 0.7)


# %% --------
df = pd.read_csv("results/defaults/p_choose_default.csv")
g = sns.FacetGrid(df, 'n_gamble', 'n_outcome', margin_titles=False)
def plot_one(data, **kws):
    sns.lineplot('cost', 'with', data=data); 
    sns.lineplot('cost', 'without', data=data, color=GRAY);
    plt.xlim(0, 0.2); plt.ylim(0, 1.02)
    plt.xticks([0, 0.1, 0.2]); plt.yticks([0, 0.5, 1])
    plt.axhline(1/data.n_gamble.iloc[0], ls='--', c='black', alpha=0.5)

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Cost')
g.set_ylabels('Prob Choose Default')
# g.set_xticklabels([0, 0.1, 0.15, 0.2])
show()

# %% ==================== Suggestion ====================

df = pd.read_csv("results/suggestions.csv")
g = sns.FacetGrid(df, 'n_gamble', 'n_outcome', margin_titles=False)
def plot_one(data, **kws):
    sns.lineplot('cost', 'with', data=data); 
    sns.lineplot('cost', 'without', data=data, color=GRAY);
    plt.xlim(0, 0.2); plt.ylim(0, 1.02)
    plt.xticks([0, 0.1, 0.2]); plt.yticks([0, 0.5, 1])
    # plt.axhline(1/data.n_gamble.iloc[0], ls='--', c='black', alpha=0.5)

g.map_dataframe(plot_one)
g.set_titles(template='{row_name} Options – {col_name} Features')
g.set_xlabels('Cost')
g.set_ylabels('Prob Choose Suggestion')
# g.set_xticklabels([0, 0.1, 0.15, 0.2])
show()