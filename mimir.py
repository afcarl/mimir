import os
import numpy as np
from sklearn import datasets, linear_model

PORTFOLIO_CUTOFF = 0.2

with open('data/full_data_assets.txt') as f0, open('data/returns.csv') as f1, open('data/value.csv') as f2:
    full_data_assets = f0.read().split()
    returns = [line.split(',') for line in f1.read().split('\n')[:10]]
    returns = [list(map(float, x[1:])) for x in returns if x[0] in full_data_assets]
    print(len(returns[0]))

    style_betas = [line.split(',') for line in f2.read().split('\n')[:10]]
    style_betas = [list(map(float, y[1:])) for y in style_betas if y[0] in full_data_assets]
    print(len(style_betas[0]))
    # print(style_betas)
    # for x in returns:
        # print(x[0])

NUM_DAYS = 80


def get_style_return(t_index):
    next_period_returns = list(zip(*returns))[t_index + 1]
    # print("NExt returns ", next_period_returns)

    ranked_indices = sorted(range(len(next_period_returns)), key=lambda asset_index: style_betas[asset_index][t_index], reverse=True)

    # print(ranked_indices)

    ranked_returns = [next_period_returns[i] for i in ranked_indices]
    # print(ranked_returns)

    top_portion = ranked_returns[:round(PORTFOLIO_CUTOFF * len(ranked_returns))]
    bottom_portion = ranked_returns[round((1 - PORTFOLIO_CUTOFF) * len(ranked_returns)):]
    # print(top_portion)
    return_to_style = sum(top_portion) - sum(bottom_portion)
    return return_to_style
    


style_returns = [get_style_return(day) for day in range(NUM_DAYS)]
print(style_returns)
print(len(style_returns))

for asset in style_betas:
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    x = np.multiply(style_returns[:NUM_DAYS], asset[:NUM_DAYS]).reshape(-1, 1)
    print(x)
    regr.fit(x, style_returns)
    print('Coefficients: \n', regr.coef_)


gearing_ratios = [1] * len(returns)
i = 0

# Initialize gearing ratios to 1

# print(full_data_assets)

## Initialize gearing ratios to 1

# start at jan 1, 2005. rank all stocks by (value * gearing ratio) 
# build portfolio with top 30% and bottom 30%
# calculate return to that portfolio over next 4 week period (may need to divide that return by n (number of stocks))
# Repeat this starting at next 4 week period, repeat to get 80 independent returns to value
# Now run regressions against each individual stock return for return to value, over 80 time periods, so we have 80 data points for regression
#    in this regression, y = return to individual stock, x = return to value * value beta

# Coefficients from said regression are the new gearing ratios, making sure to exclude (set gearing ratio to zero) those stocks where the error term for that coefficient is above certain threshold
# Now go back to beginning of process and start again with these new gearing ratios
# repeat until gearing ratios converge

