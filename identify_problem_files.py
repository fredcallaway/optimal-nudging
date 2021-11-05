import pandas as pd
import numpy as np
import json

def get_points_click_cost(row_number,click_cost,raw_cost_matrix):
    # click cost from cost matrix
    parsed_cost_matrix = np.array(json.loads(raw_cost_matrix)).flatten()
    click_cost_from_matrix = (parsed_cost_matrix.shape[0]*click_cost) - np.sum(parsed_cost_matrix)
    # click cost from click values
    return click_cost_from_matrix

def get_points_action_utility(raw_payoff_matrix,raw_weights,selected_option):
    parsed_weights = np.array(json.loads(raw_weights)).flatten()
    parsed_payoff_matrix = np.array(json.loads(raw_payoff_matrix))
    prize_counts = parsed_payoff_matrix[:,selected_option]
    return np.inner(parsed_weights,prize_counts)

data = pd.read_csv('data/final_experiments_data/supersize_data.csv')
print(data.shape)

# index_list = []
# first_iteration = True
# for index,row in data.iterrows():
#     points_click_cost = get_points_click_cost(index,row['cost'],row['cost_matrix'])
#     points_action_utility = get_points_action_utility(row['payoff_matrix'],row['weights'], row['selected_option'])
#     points_metalevel_reward = points_action_utility - points_click_cost
#     difference = abs(points_metalevel_reward - (row['net_earnings']*3000))
#     if (difference > 1.1):
#         index_list.append(index)

# problem_df = data.iloc[index_list]
# problem_df.to_csv('problem_data.csv')


