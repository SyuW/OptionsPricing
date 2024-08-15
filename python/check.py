from numpy import sqrt

from pricers import blackScholesPricer, binomialTreePricer, finiteDifferencesPricer, monteCarloPricer
from utils import getInput, interpolateOptionPrices


def checkConsistencyOfMethods():
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

    print(f"The Black-Scholes-Merton price for the option is: {bsm_price}.")
    print(f"The binomial tree price for the option is: {bin_price}.")
    print(f"The finite differences price for the option is {fd_price}.")
    print(f"The Monte Carlo price for the option is {mc_price} with a 95% confidence interval of [{ci_lower}, {ci_upper}].")


if __name__ == "__main__":
    pass