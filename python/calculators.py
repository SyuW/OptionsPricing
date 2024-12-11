import numpy as np
from numpy import sqrt, exp, array, arange, log
import matplotlib.pyplot as plt


# Black-Scholes analytical pricing formula for european options
def blackScholesPricer(S, K, r, sigma, T, q, type="call"):
    """
    Pricer for european options using the classic Black-Scholes-Merton formulas

    S       : stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    q       : continuous dividend yield
    """

    assert type in ["call", "put"]

    d_1 = (log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d_2 = (log(S / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * sqrt(T))

    if type == "call":
        option_price = S * exp(-q * T) * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2)
    elif type == "put":
        option_price = K * exp(-r * T) * norm.cdf(-d_2) - S * exp(-q * T) * norm.cdf(-d_1)

    return option_price


def getBlackScholesGreeks(S, K, r, sigma, T, q, greek, type="call"):
    """
    Compute the Greek letters for European options using the analytical
    Black-Scholes-Merton formulas

    S       : stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to maturity
    q       : continuous dividend yield
    """

    assert type in ["call", "put"]
    assert greek in ["delta", "gamma", "theta", "vega", "rho"]

    d_1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d_2 = (log(S / K) + (r - sigma ** 2 / 2) * T) / (sigma * sqrt(T))

    if type == "call":

        if greek == "delta":
            return exp(-q * T) * norm.cdf(d_1)

        elif greek == "gamma":
            return (norm.pdf(d_1) * exp(-q * T)) / (S * sigma * sqrt(T))

        elif greek == "theta":
            return -S * norm.pdf(d_1) * sigma * exp(-q * T) / (2 * sqrt(T)) \
                   + q * S * norm.cdf(d_1) * exp(-q * T) \
                   - r * K * exp(-r * T) * norm.cdf(d_2)

        elif greek == "vega":
            return S * sqrt(T) * norm.pdf(d_1) * exp(-q * T)
        
        elif greek == "rho":
            return K * T * exp(-r * T) * norm.cdf(d_2)

    elif type == "put":

        if greek == "delta":
            return exp(-q * T) * (norm.cdf(d_1) - 1)

        elif greek == "gamma":
            return (norm.pdf(d_1) * exp(-q * T)) / (S * sigma * sqrt(T))

        elif greek == "theta":
            return (-S * norm.pdf(d_1) * sigma * exp(-q * T)) / (2 * sqrt(T)) \
                    - q * S * norm.cdf(-d_1) * exp(-q * T) \
                    + r * K * exp(-r * T) * norm.cdf(-d_2)

        elif greek == "vega":
            return S * sqrt(T) * norm.pdf(d_1) * exp(-q * T)

        elif greek == "rho":
            return -K * T * exp(-r * T) * norm.cdf(-d_2)


def asianOptionPricerExact(S, K, r, sigma, T, type="call"):
    """
    Pricer for geometric Asian options (without early exercise) using exact analytical expressions

    S       : stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    """

    assert type in ["call", "put"]

    sigma_G = sigma / sqrt(3)
    b = 0.5 * (r - 0.5 * sigma_G ** 2)
    d_1 = (log(S / K) + (b + 0.5 * sigma_G ** 2) * T) / (sigma_G * sqrt(T))
    d_2 = (log(S / K) + (b - 0.5 * sigma_G ** 2) * T) / (sigma_G * sqrt(T))

    if type == "call":
        option_price = S * exp((b-r) * T) * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2)
    elif type == "put":
        option_price = K * exp(-r * T) * norm.cdf(-d_2) - S * exp((b-r) * T) * norm.cdf(-d_1)

    return option_price