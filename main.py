from pricers import blackScholesPricer, finiteDifferencesPricer
import matplotlib.pyplot as plt

from numpy import exp


if __name__ == "__main__":
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
