def load_data(name=''):
    df = pd.read_csv(f'../data/{name}.csv')
    df.rename(columns={
        'cost': 'reveal_cost', 
        'points_click_cost': 'decision_cost',
        'chose_nudge': 'choose_default',
        'og_baskets': 'n_option',
        'num_features': 'n_feature',
        'weights_deviation': 'weight_dev',
        'points_action_utility': 'payoff',
        'net_earnings': 'meta_return'
    }, inplace=True)
    df['meta_return'] =  df.payoff - df.decision_cost
    # df['choose_immediately'] = df.click_cost <= 1e-7
    # df['nudge'] = (df.trial_nudge == 'default').astype(int)
    return df

data = {
    'model': load_sim('attention_sims_alt'),
    'human': load_data('final_experiments_data/attention_experiment_fixed')
}

# %% --------

# def relative(x, lo, hi):
#     return (x - lo) / (hi - lo)

# def get_relative(row, var):
#     payoffs = np.array(json.loads(row.weights)) @ np.array(json.loads(row.payoff_matrix))
#     return relative(row[var], payoffs.mean(), payoffs.max())

# df['relative_payoff'] = df.apply(get_relative, var='payoff', axis=1)
# df['relative_meta_return'] = df.apply(get_relative, var='meta_return', axis=1)

# %% --------
random_payoff = 150
maximum_payoff = 183.63861

def relative(x):
    return (x - random_payoff) / (maximum_payoff - random_payoff)

for df in data.values():
    df['relative_payoff'] = relative(df.payoff)
    df['relative_meta_return'] = relative(df.meta_return)

# %% --------
df = data['model']

lblue, dblue, _, _, lred, dred, *_ = sns.color_palette('Paired')
lgray = (0.7, 0.7, 0.7)
dgray = (0.5, 0.5, 0.5)

def attention_performance(df, agent):
    ci = 0.95 if agent == 'human' else False
    sns.barplot('nudge_type', 'payoff', data=df, 
        errcolor='.46',
        palette={'random': lgray, 'extreme': lred, 'greedy': lblue})
    sns.barplot('nudge_type', 'meta_return', data=df,
        palette={'random': dgray, 'extreme': dred, 'greedy': dblue})
    plt.ylabel('Points')
    plt.xticks(range(3), labels=['Random', 'Extreme', 'Optimal'])
    plt.xlabel('')
    plt.axhline(maximum_payoff, ls='--', color='k')
    plt.ylim(random_payoff, 185)

plot_both(attention_performance, data)
# %% --------


# def attention_performance(df, agent):
#     ci = 0.95 if agent == 'human' else False
#     sns.barplot('nudge_type', 'relative_payoff', data=df, 
#         errcolor='.46',
#         palette={'random': lgray, 'extreme': lred, 'greedy': lblue})
#     sns.barplot('nudge_type', 'relative_meta_return', data=df,
#         palette={'random': dgray, 'extreme': dred, 'greedy': dblue})
#     plt.ylabel('Relative Utility')
#     plt.xticks(range(3), labels=['Random', 'Extreme', 'Optimal'])
#     plt.xlabel('')
#     plt.ylim(0, 1)

# X = pd.melt(df, id_vars='nudge_type', value_vars=['relative_payoff', 'relative_meta_return'])
# p = sns.barplot('nudge_type', 'value', hue='variable', data=X)
# plt.legend().remove()
# show()
# # %% --------
