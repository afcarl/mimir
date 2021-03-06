import numpy as np
import statsmodels.api as sm

# Parameters

PORTFOLIO_CUTOFF = 0.2
NUM_T_PERIODS = 80
NUM_ASSETS = 1000
DELTA_THRESHOLD = 0.01
ERROR_THRESHOLD = 0.1

# Read data from files
with open('data/full_data_assets.txt') as f0, open('data/returns.csv') as f1, open('data/value.csv') as f2:
    full_data_assets = f0.read().split()
    unfiltered_returns = [line.split(',') for line in f1.read().split('\n')]

    # Filter for data for which we have a complete set for
    filtered_data = [x for x in unfiltered_returns if x[0] in full_data_assets]

    # The first column is the asset ids, so strip them off an just get returns
    returns = [list(map(float, x[1:])) for x in filtered_data]

    asset_ids = list(zip(*filtered_data))[0]

    style_betas = [line.split(',') for line in f2.read().split('\n')]
    style_betas = [list(map(float, y[1:])) for y in style_betas if y[0] in full_data_assets]

    assert len(returns) == len(style_betas), "Num returns does not match num style betas"
    assert len(returns) == len(asset_ids), "Num returns does not match num asset ids"
    assert len(returns[0]) == len(style_betas[0]), "Time period mismatch between returns and betas"


# Function to calculate the return to style over a given (t_index) 4-week period
def get_style_return(t_index, all_asset_returns, all_asset_style_betas, gearing_ratios):
    # We are using the return from the t+1 time period, with the beta from the t time period
    # to calculate a factor return for the t+1 time period
    next_period_returns = list(zip(*all_asset_returns))[t_index + 1]

    # We rank returns in descending order based on their (gearing ratio * style beta) value
    cmp_function = lambda asset_index: gearing_ratios[asset_index] * all_asset_style_betas[asset_index][t_index]
    ranked_indices = sorted(range(len(next_period_returns)), key=cmp_function, reverse=True)

    ranked_returns = [next_period_returns[i] for i in ranked_indices]

    # Our portfolio for returns to style is 'buying' the top ranked assets, and shorting the bottom ranked assets
    top_portion = ranked_returns[:round(PORTFOLIO_CUTOFF * len(ranked_returns))]
    bottom_portion = ranked_returns[round((1 - PORTFOLIO_CUTOFF) * len(ranked_returns)):]

    return_to_style = sum(top_portion) - sum(bottom_portion)
    return return_to_style
    

def get_gearing_ratio(asset_style_betas, asset_returns, style_returns):
    # This is a linear regression with one feature.
    # The independent variable X is the style beta multiplied by return to style.
    x = np.multiply(asset_style_betas[:NUM_T_PERIODS], style_returns[:NUM_T_PERIODS]).reshape(-1, 1)

    # Add a constant term since data is not necessarily centered. Do we need this?
    x = sm.add_constant(x)

    # The dependent variable Y is the return to the asset.
    # We are using the asset return from the t+1 time period. Factor return is also
    # from t+1 time period, but the array starts there anyway so no need for +1 in style_returns
    results = sm.OLS(asset_returns[1:NUM_T_PERIODS + 1], x).fit()

    coefficient = results.params[1]
    standard_error = results.bse[1]

    if standard_error == 0 and coefficient == 0:
        return 0

    t_statistic = coefficient / standard_error

    # If t_statistic for the coefficient is below a certain threshold, meaning we are not sufficiently
    # confident about the estimation of that coefficient, then we set it to zero, which hopefully
    # excludes it from the style portfolio in next iteration
    if t_statistic < ERROR_THRESHOLD:
        return 0

    # The estimated coefficient is the gearing ratio
    return coefficient


# Initialize gearing ratios to 1
gearing_ratios = [1] * len(asset_ids)
delta = DELTA_THRESHOLD
old_style_returns = None

# Continue running this exercise until the change in gearing ratios falls below a certain threshold
while delta >= DELTA_THRESHOLD:

    new_style_returns = [get_style_return(t, returns, style_betas, gearing_ratios) for t in range(NUM_T_PERIODS)]

    # Compute sum of squared differences between old style returns and new ones
    delta = sum(np.subtract(new_style_returns, old_style_returns) ** 2) if old_style_returns != None else DELTA_THRESHOLD
    print("Delta ", delta)

    gearing_ratios = [get_gearing_ratio(style_betas[i], returns[i], new_style_returns) for i in range(len(returns))]

    old_style_returns = new_style_returns
