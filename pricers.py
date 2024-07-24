import numpy as np
from numpy import sqrt, exp, array, arange, log
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy.interpolate import CubicSpline


# Black-Scholes analytical pricing formula for european options
def blackScholesPricer(S, K, r, sigma, T, type="call"):
    """
    Pricer for european options using the classic Black-Scholes-Merton formulas

    S       : stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    """

    assert type in ["call", "put"]

    d_1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d_2 = (log(S / K) + (r - sigma ** 2 / 2) * T) / (sigma * sqrt(T))

    if type == "call":
        option_price = S * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2)
    elif type == "put":
        option_price = K * exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    return option_price


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


def binomialPricer(S, K, r, sigma, T, q, n, type="call", style="european", visualize=False):
    """
    Cox-Ross-Rubinstein (CRR) Binomial tree pricer

    S       : initial stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    q       : continuous dividend yield
    n       : size of tree
    """

    assert type in ["call", "put"]
    assert style in ["american", "european"]

    # Useful quantities to define
    t = T / n
    u = exp(sigma * sqrt(t))
    d = exp(-sigma * sqrt(t))
    a = exp((r-q) * t)
    p = (a - d) / (u - d)

    price_tree = []
    for i in range(0, n+1):
        node_list = np.zeros((i+1,2)) # left to right encodes lowest to highest
        for j in range(0, i+1):
            # each node is [stock price, option price]: option price is zero-initialized
            node_list[j, 0] = S * (u ** j) * (d ** (i-j))
        price_tree.append(node_list)

    # calculate option prices at expiration date
    for j in range(0, n+1):
        S_nj = price_tree[-1][j, 0]

        if type == "call":
            price_tree[-1][j, 1] = max(S_nj - K, 0)
        elif type == "put":
            price_tree[-1][j, 1] = max(K - S_nj, 0) 

    # update option prices but moving backwards through the tree
    for i in range(n-1, -1, -1):
        for j in range(0, i+1):

            # discounted expected value
            binomial_price = exp(-r * t) * (p * price_tree[i+1][j+1, 1] + (1-p) * price_tree[i+1][j, 1])

            S_ij = price_tree[i][j, 0]
            if type == "call":
                exercise_price = max(S_ij - K, 0)
            elif type == "put":
                exercise_price = max(K - S_ij, 0) 
            
            if style == "american":
                price_tree[i][j, 1] = max(binomial_price, exercise_price)
            elif style == "european":
                price_tree[i][j, 1] = binomial_price

    if visualize:
        for i in range(0, n+1):
            print(price_tree[i])

    return price_tree[0][0, 1]


def trinomialPricer(S, K, r, sigma, T, q, n, type="call", style="european", visualize=False):
    """
    Implement trinomial tree pricer, to compare with explicit finite differences method.
    Learn about recombination in the tree?

    """
    


    return


# use finite differences to solve the Black-Scholes PDE to obtain option price
def finiteDifferencesPricer(K, r, sigma, q, S_max, M, T, N, type="call", style="european", version="implicit", use_log_price=False):
    """
    Finite differences pricer (implicit version)

    K : strike price
    r : risk-free interest rate
    sigma : volatility
    S_max: maximum stock price
    T : total time until maturity (in years)
    q : continuous dividend yield
    N : number of grid-points in time
    M : number of grid-points in stock price
    """

    assert version in ["implicit", "explicit", "crank_nicolson"]
    assert style in ["european", "american"]
    assert type in ["call", "put"]

    deltaT = T / N
    deltaS = S_max / M

    # stock price increases from left-to-right: 0, deltaS, 2 * deltaS, ..., (M-1) * deltaS, S_max
    # time increases from top-to-bottom: 0, deltaT, 2 * deltaT, ..., (N-1) * deltaT, T
    grid = np.zeros((N+1, M+1)) # grid is (N + 1) x (M + 1): time on vertical, stock price on horizontal

    # helper arrays
    t_arr = arange(0, N+1) * deltaT
    S_arr = arange(0, M+1) * deltaS

    # boundary conditions on domain: [0, T] x [0, S_max]
    if style == "european":

        if type == "put":
            zero_S_bc = K * exp(-r * (T - t_arr))
            large_S_bc = 0 # the no-arbitrage bound is p >= max(Ke^{-rT}-S_0, 0), but we choose S_max large enough
            maturity_bc = np.maximum(K - S_arr, 0)

        elif type == "call":
            zero_S_bc = 0
            large_S_bc = S_max - K * exp(-r * (T - t_arr))
            maturity_bc = np.maximum(S_arr - K, 0)

    elif style == "american":

        if type == "put":
            zero_S_bc = K
            large_S_bc = 0
            maturity_bc = np.maximum(K - S_arr, 0)
            
        elif type == "call":
            zero_S_bc = 0
            large_S_bc = S_max - K
            maturity_bc = np.maximum(S_arr - K, 0)

    # apply boundary conditions to grid
    grid[:, -1] = large_S_bc
    grid[:, 0] = zero_S_bc
    grid[-1, :] = maturity_bc

    if version == "implicit":

        j_arr = arange(1, M)

        # coefficients
        a_vec = +0.5 * (r - q) * j_arr * deltaT - 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT
        b_vec = 1 + r * deltaT + (sigma ** 2) * (j_arr ** 2) * deltaT
        c_vec = -0.5 * (r - q) * j_arr * deltaT - 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT

        # need to solve a sparse linear system for each iteration
        tri = csc_matrix(np.diag(a_vec[1:], k=-1) + np.diag(b_vec, k=0) + np.diag(c_vec[:-1], k=1))

        # backwards iteration
        for i in range(N-1, -1, -1):

            # offset due to boundary conditions
            offset = np.zeros(M-1)
            offset[0] = a_vec[0] * grid[i, 0]
            offset[-1] = c_vec[-1] * grid[i, M]
            
            forward = grid[i+1, 1:M]

            prices_at_iteration = spsolve(tri, forward - offset)

            if style == "european":
                pass

            elif style == "american":
                if type == "put":
                    prices_at_iteration = np.maximum(prices_at_iteration, K - arange(1, M) * deltaS)
                elif type == "call":
                    prices_at_iteration = np.maximum(prices_at_iteration, arange(1, M) * deltaS - K)

            grid[i, 1:M] = prices_at_iteration

    # Broken right now ... trying to fix
    elif version == "explicit":

        j_arr = arange(1, M)

        # coefficients
        a_vec = -0.5 * (r-q) * j_arr * deltaT + 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT # for f_{i, j-1}
        b_vec = 1 - r * deltaT - (sigma ** 2) * (j_arr ** 2) * deltaT                      # for f_{i, j} 
        c_vec = +0.5 * (r-q) * j_arr * deltaT + 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT # for f_{i, j+1}

        # tridiagonal matrix
        tri = csc_matrix(np.diag(a_vec[1:], k=-1) + np.diag(b_vec, k=0) + np.diag(c_vec[:-1], k=1))

        # backwards iteration
        for i in range(N-1, -1, -1):
            
            forward = grid[i+1, 1:M]

            # offset due to boundary conditions
            offset = np.zeros(M-1)
            offset[0] = a_vec[0] * grid[i, 0]
            offset[-1] = c_vec[-1] * grid[i, M]
            
            prices_at_iteration = tri.dot(forward) + offset

            grid[i, 1:M] = prices_at_iteration

    return arange(0, M+1) * deltaS, grid[0, :]


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
    T_sqrt = sqrt(T)

    # volatility and interest rates are constant
    rng = np.random.default_rng(300)

    if style == "european":

        # pricing func
        pricing_func = lambda eps: S_0 * exp((r - sigma ** 2 / 2) * T + sigma * T_sqrt * eps)
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
        times = arange(0, T, deltaT)
        variates = rng.standard_normal((N, num_trials))
        variates[0, :] = 0

        # since the stock price follows geometric brownian motion, it is lognormally distributed
        log_S_t = log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(variates, axis=0)

        # geometric average of stock price
        S_ave = exp((1/T) * np.sum(log_S_t, axis=0) * deltaT)

        if type == "call":

            if method == "default":
                
                sampled_payoffs = np.maximum(S_ave - K, 0) 

            elif method == "antithetic":
                
                anti = log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(-variates, axis=0)
                S_ave_anti = exp((1/T) * np.sum(anti, axis=0) * deltaT)

                sampled_payoffs = (np.maximum(S_ave - K, 0) + np.maximum(S_ave_anti - K, 0)) / 2

        elif type == "put":

            if method == "default":

                sampled_payoffs = np.maximum(K - S_ave, 0) 

            elif method == "antithetic":

                anti = log(S_0) + ((r - sigma ** 2 / 2) * times)[:, np.newaxis] + sigma * deltaT_sqrt * np.cumsum(-variates, axis=0)
                S_ave_anti = exp((1/T) * np.sum(anti, axis=0) * deltaT)

                sampled_payoffs = (np.maximum(K - S_ave, 0) + np.maximum(K - S_ave_anti, 0)) / 2

    return exp(-r*T) * np.mean(sampled_payoffs), np.std(exp(-r*T) * sampled_payoffs)