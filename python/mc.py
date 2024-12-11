import numpy as np


def monteCarloPricer(S_0, K, r, sigma, q, T, N, num_trials, style="european", type="put", method="default"):
    """
    Monte Carlo pricer

    How it works:
    - sample a path for S(t) in a risk neutral world
    - calculate the payoff from the derivative
    - repeat steps 1 and 2
    - calculate the mean of sample payoffs to get estimate of the expected payoff
    - discount expected payoff at the risk free rate
    """

    assert type in ["call", "put"]
    assert style in ["european", "american", "geometric_asian", "arithmetic_asian"]
    assert method in ["default", "antithetic", "importance"]

    deltaT = T / N
    deltaT_sqrt = np.sqrt(T/N)
    T_sqrt = np.sqrt(T)

    # volatility and interest rates are constant
    rng = np.random.default_rng(300)

    if style == "european":

        # pricing func
        pricing_func = lambda eps: S_0 * np.exp((r - sigma ** 2 / 2) * T + sigma * T_sqrt * eps)
        variates = rng.standard_normal(num_trials)
        
        if type == "call":

            if method == "default":
                sampled_payoffs = np.maximum(pricing_func(variates) - K, 0)

            elif method == "antithetic":
                f1 = np.maximum(pricing_func(variates) - K, 0)
                f2 = np.maximum(pricing_func(-variates) - K, 0)
                sampled_payoffs = (f1 + f2) / 2

        elif type == "put":

            if method == "default":
                sampled_payoffs = np.maximum(K - pricing_func(variates), 0)
            
            elif method == "antithetic":
                f1 = np.maximum(K - pricing_func(variates), 0)
                f2 = np.maximum(K - pricing_func(-variates), 0)
                sampled_payoffs = (f1 + f2) / 2

    elif style == "arithmetic_asian":

        pass

    elif style == "geometric_asian":

        # need to generate price paths
        times = np.arange(0, T, deltaT)
        variates = rng.standard_normal((N, num_trials))
        variates[0, :] = 0

        # since the stock price follows geometric brownian motion, it is lognormally distributed
        log_S_t = np.log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(variates, axis=0)

        # geometric average of stock price
        S_ave = np.exp((1/T) * np.sum(log_S_t, axis=0) * deltaT)

        if type == "call":

            if method == "default":
                
                sampled_payoffs = np.maximum(S_ave - K, 0) 

            elif method == "antithetic":
                
                anti = np.log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(-variates, axis=0)
                S_ave_anti = np.exp((1/T) * np.sum(anti, axis=0) * deltaT)

                sampled_payoffs = (np.maximum(S_ave - K, 0) + np.maximum(S_ave_anti - K, 0)) / 2

        elif type == "put":

            if method == "default":

                sampled_payoffs = np.maximum(K - S_ave, 0) 

            elif method == "antithetic":

                anti = np.log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(-variates, axis=0)
                S_ave_anti = np.exp((1/T) * np.sum(anti, axis=0) * deltaT)

                sampled_payoffs = (np.maximum(K - S_ave, 0) + np.maximum(K - S_ave_anti, 0)) / 2

    return np.exp(-r*T) * np.mean(sampled_payoffs), np.std(np.exp(-r*T) * sampled_payoffs)