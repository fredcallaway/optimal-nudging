import pandas as pd
import numpy as np
import json
import sys
# file = '../data/pilot_data/traffic-light-pilot4.csv'
file = sys.argv[1]
df = pd.read_csv(file)
# %% --------
def max_highlight_value(row):
    X = np.array(json.loads(row.payoff_matrix))
    x = X[row.highlight_index]

    return x.max()

df['max_highlight_value'] = df.apply(max_highlight_value, axis=1)
df.to_csv(file, index=False)
