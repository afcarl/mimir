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
    returns = [line.split(',') for line in f1.read().split('\n')[:NUM_ASSETS]]
    returns = [list(map(float, x[1:])) for x in returns if x[0] in full_data_assets]

    style_betas = [line.split(',') for line in f2.read().split('\n')[:NUM_ASSETS]]
    style_betas = [list(map(float, y[1:])) for y in style_betas if y[0] in full_data_assets]

    assert len(returns) == len(style_betas), "Num returns does not match num style betas"
    # FIXME: add this back once we have data for last 4-week return
    # assert len(returns[0]) == len(style_betas[0]), "Time period mismatch between returns and betas"


# Function to calculate the return to style over a given (t_index) 4-week period
def get_style_return(t_index, all_asset_returns, all_asset_style_betas, gearing_ratios):
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

    # The dependent variable Y is the return to the asset
    results = sm.OLS(asset_returns[:NUM_T_PERIODS], x).fit()

    coefficient = results.params[1]
    standard_error = results.bse[1]

    # If error term for the coefficient is above a certain threshold, then we set the coefficient
    # to zero, which hopefully excludes it from the style portfolio in next iteration
    if standard_error > ERROR_THRESHOLD:
        coefficient = 0

    # The estimated coefficient is the gearing ratio
    return coefficient


# Initialize gearing ratios to 1
gearing_ratios = [1] * len(returns)
delta = DELTA_THRESHOLD

# Continue running this exercise until the change in gearing ratios falls below a certain threshold
while delta >= DELTA_THRESHOLD:

    style_returns = [get_style_return(t, returns, style_betas, gearing_ratios) for t in range(NUM_T_PERIODS)]

    new_gearing_ratios = [get_gearing_ratio(style_betas[i], returns[i], style_returns) for i in range(len(returns))]

    # Compute sum of squared differences between old gearing ratios and new ones
    delta = sum(np.subtract(new_gearing_ratios, gearing_ratios) ** 2)
    print("Delta ", delta)

    # Update gearing ratios for next iteration
    gearing_ratios = new_gearing_ratios

