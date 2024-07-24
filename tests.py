from pricers import asianOptionPricerExact, blackScholesPricer, finiteDifferencesPricer, monteCarloPricer
import matplotlib.pyplot as plt

from numpy import exp, sqrt


def testMonteCarlo():

    initial_S = 100
    strike = 95
    interest_rate = 0.01
    maturity = 0.25
    volatility = 0.50
    dividend_rate = 0

    trials = 1000000

    mc_price, mc_std = monteCarloPricer(S_0=initial_S, K=strike, r=interest_rate, sigma=volatility, q=dividend_rate,
                                    T=maturity, N=1000, num_trials=trials,
                                    style="geometric_asian", type="put", method="antithetic")
    
    exact_price = asianOptionPricerExact(S=initial_S, K=strike, r=interest_rate, sigma=volatility, T=maturity, type="put")

    print("The exact price is:", exact_price)
    print("The Monte Carlo price is:", mc_price)
    print("The standard deviation for payoffs is:", mc_std)
    print("A 95 percent confidence interval for the option price is:", [mc_price - 1.96 * mc_std / sqrt(trials), mc_price + 1.96 * mc_std / sqrt(trials)])

    return


def testFiniteDifferences():
    strike = 50
    interest_rate = 0.10
    volatility = 0.40
    dividend_rate = 0
    maturity = 0.4167

    s, sol = finiteDifferencesPricer(K=strike, r=interest_rate, sigma=volatility, q=dividend_rate,
                                    S_max=400, M=1500, T=maturity, N=1500,
                                    type="call", style="european")

    # true solution comes Black-Scholes-Merton formula
    true_sol = blackScholesPricer(s[1:], K=strike, r=interest_rate, sigma=0.40, T=maturity, type="call")

    plt.xlabel(r"Initial stock price, $S_0$")
    plt.ylabel(r"Put option price, $p$")

    plt.plot(s, sol, linewidth=3, label="Finite differences")
    plt.plot(s[1:], true_sol, linewidth=3, label="Ground truth")
    plt.plot(s, s - strike * exp(-interest_rate * maturity),
            linestyle="--", alpha=0.6, color="black", label=r"$S_0-Ke^{-rT}$")
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()


if __name__ == "__main__":
    
    test_params_1 = {
        "strike": 50,
        "interest_rate": 0.10,
        "volatility": 0.40,
        "dividend_rate": 0,
        "maturity": 0.4167
    }
    
    test_params_2 = {
        "initial_stock_price": 100,
        "strike": 95,
        "interest_rate": 0.01,
        "maturity": 0.25,
        "volatility": 0.50,
        "dividend_rate": 0
    }


    testMonteCarlo()