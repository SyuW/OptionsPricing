from pricers import asianOptionPricerExact, binomialTreePricer, blackScholesPricer, getBlackScholesGreeks, \
                    finiteDifferencesPricer, monteCarloPricer

from numpy import sqrt
from utils import interpolateOptionPrices, getInput

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import argparse
import time


## Script for freely using any of the pricing engines through command line arguments
if __name__ == "__main__":

    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    option_type = getInput(message="What type of option are considering? Enter it here: ",
                            assert_list=["call", "put"], func=str)
    stock_price = getInput(message="What is the current stock price? Enter it here: ")
    strike_price = getInput(message="What is the option's strike price? Enter it here: ")
    interest_rate = getInput(message="What is the interest rate? Enter it here: ")
    volatility = getInput(message="What is the volatility? Enter it here: ")
    maturity = getInput(message="What is the time until maturity? Enter it here: ")

    bsm_price = blackScholesPricer(S=stock_price, K=strike_price, r=interest_rate,
                                   sigma=volatility, T=maturity, q=0, type=option_type)
    bin_price = binomialTreePricer(S=stock_price, K=strike_price, r=interest_rate, 
                               sigma=volatility, T=maturity, q=0, n=1000, type=option_type)
    
    fd_price = interpolateOptionPrices(stock_price, *finiteDifferencesPricer(K=strike_price, r=interest_rate, sigma=volatility, q=0,
                                                                             S_max=2 * stock_price, M=200, N="auto", T=maturity, type=option_type,
                                                                             version="explicit"))
    
    trials = 10000000
    mc_price, mc_std = monteCarloPricer(S_0=stock_price, K=strike_price, r=interest_rate,
                                        sigma=volatility, q=0, T=maturity, N=1000, num_trials=trials, type=option_type, method="antithetic")
    ci_lower = mc_price - 1.96 * mc_std / sqrt(trials)
    ci_upper = mc_price + 1.96 * mc_std / sqrt(trials)

    # find the Greeks
    delta = getBlackScholesGreeks(S=stock_price, K=strike_price, r=interest_rate, sigma=volatility, T=maturity, q=0, greek="delta", type=option_type)
    gamma = getBlackScholesGreeks(S=stock_price, K=strike_price, r=interest_rate, sigma=volatility, T=maturity, q=0, greek="gamma", type=option_type)
    theta = getBlackScholesGreeks(S=stock_price, K=strike_price, r=interest_rate, sigma=volatility, T=maturity, q=0, greek="theta", type=option_type)
    vega = getBlackScholesGreeks(S=stock_price, K=strike_price, r=interest_rate, sigma=volatility, T=maturity, q=0, greek="vega", type=option_type)
    rho = getBlackScholesGreeks(S=stock_price, K=strike_price, r=interest_rate, sigma=volatility, T=maturity, q=0, greek="rho", type=option_type)
    
    print("-"*50)
    print("OUTPUTS")
    print("-"*50)

    print("*")
    print("*")
    print("*")
    print("-"*50)
    print("Prices")
    print("-"*50)
    print(f"The Black-Scholes-Merton price for the option is: {bsm_price:.4f}.")
    print(f"The binomial tree price for the option is: {bin_price:.4f}.")
    print(f"The finite differences price for the option is {fd_price:.4f}.")
    print(f"The Monte Carlo price for the option is {mc_price:.4f} with a 95% confidence interval of [{ci_lower:.4f}, {ci_upper:.4f}].")
    print("-"*50)

    print("*")
    print("*")
    print("*")
    print("-"*50)
    print("Greeks")
    print("-"*50)
    print(f"Delta: {delta:.6f}")
    print(f"Gamma: {gamma:.6f}")
    print(f"Theta: {theta:.6f}")
    print(f"Theta (per calendar day): {(theta / 365):.6f}")
    print(f"Theta (per trading day): {(theta / 252):.6f}")
    print(f"Vega: {vega:.6f}")
    print(f"Vega (one percent change in volatility): {(vega / 100):.6f}")
    print(f"Rho:", f"{rho:.6f}")
    print(f"Rho (one percent change in interest rate):", f"{(rho / 100):.6f}")
    print("-"*50)