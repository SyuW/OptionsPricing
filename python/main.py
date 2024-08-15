from pricers import asianOptionPricerExact, binomialTreePricer, blackScholesPricer, \
                    finiteDifferencesPricer, monteCarloPricer

from numpy import sqrt
from utils import interpolateOptionPrices

from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import argparse
import time


def get_input(message, assert_list=[], func=float):

    flag = False
    while not flag:
        try:
            user_input = input(message)
            user_input = func(user_input)
            flag = True
        except ValueError:
            print("Error: input could not be converted, please try again.")
        except Exception as e:
            print("An error occurred:", str(e), ". Please try again.")

    # if user input is a string, need to check that string is valid for later
    if func == str:
        assert user_input in assert_list

    return func(user_input)


## Script for freely using any of the pricing engines through command line arguments
if __name__ == "__main__":

    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    option_type = get_input(message="What type of option are considering? Enter it here: ",
                            assert_list=["call", "put"], func=str)
    stock_price = get_input(message="What is the current stock price? Enter it here: ")
    strike_price = get_input(message="What is the option's strike price? Enter it here: ")
    interest_rate = get_input(message="What is the interest rate? Enter it here: ")
    volatility = get_input(message="What is the volatility? Enter it here: ")
    maturity = get_input(message="What is the time until maturity? Enter it here: ")

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

    print(f"The Black-Scholes-Merton price for the option is: {bsm_price}.")
    print(f"The binomial tree price for the option is: {bin_price}.")
    print(f"The finite differences price for the option is {fd_price}.")
    print(f"The Monte Carlo price for the option is {mc_price} with a 95% confidence interval of [{ci_lower}, {ci_upper}].")