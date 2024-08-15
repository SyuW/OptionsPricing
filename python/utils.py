import numpy as np
from scipy.interpolate import CubicSpline

def pnorm_error():

    return


def interpolateOptionPrices(spot, stock_prices, option_prices):
    
    cs = CubicSpline(stock_prices, option_prices)

    return cs(spot)