import numpy as np
from numpy import sqrt, exp, array, arange
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy.interpolate import CubicSpline


# Black-Scholes analytical pricing formula for european options
def blackScholesPricer(S, K, r, sigma, T, type="call"):
    """
    S       : stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    """

    assert type in ["call", "put"]

    d_1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d_2 = (np.log(S / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    if type == "call":
        option_price = S * norm.cdf(d_1) - K * exp(-r * T) * norm.cdf(d_2)
    elif type == "put":
        option_price = K * exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

    return option_price


# Cox-Ross-Rubinstein (CRR) Binomial tree pricer
def binomialPricer(S, K, r, sigma, T, q, n, type="call", style="european", visualize=False):
    """
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


# use finite differences to solve the Black-Scholes PDE to obtain option price
def finiteDifferencesPricer(K, r, sigma, q, S_max, M, T, N, type="call", style="european"):
    """
    K : strike price
    r : risk-free interest rate
    sigma : volatility
    S_max: maximum stock price
    T : total time until maturity (in years)
    q : continuous dividend yield
    N : number of grid-points in time
    M : number of grid-points in stock price
    """

    assert style in ["european", "american"]
    assert type in ["call", "put"]

    deltaT = T / N
    deltaS = S_max / M

    # stock price increases from top-to-bottom: 0, deltaS, 2 * deltaS, ..., (M-1) * deltaS, S_max
    # time increases from left-to-right: 0, deltaT, 2 * deltaT, ..., (N-1) * deltaT, T
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

    # display_matrix(grid)

    # coefficients
    a_vec = 0.5 * (r - q) * arange(1, M) * deltaT - 0.5 * (sigma ** 2) * (arange(1, M) ** 2) * deltaT
    b_vec = 1 + r * deltaT + (sigma ** 2) * (arange(1, M) ** 2) * deltaT
    c_vec = -0.5 * (r - q) * arange(1, M) * deltaT - 0.5 * (sigma ** 2) * (arange(1, M) ** 2) * deltaT

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

    # grid = np.matrix.round(grid, decimals=2); display_matrix(grid)
    # plt.plot(np.arange(0, S_max + deltaS, deltaS), grid[:, 0])
    # plt.plot(np.arange(0, S_max + deltaS, deltaS), np.arange(0, S_max + deltaS, deltaS) - K * exp(-r * T))

    return arange(0, M+1) * deltaS, grid[0, :]


# use Monte Carlo methods for pricing the option
def monteCarloPricer(K, r, sigma, q):

    # sample a path for S(t) in a risk neutral world
    # calculate the payoff from the derivative
    # repeat steps 1 and 2
    # calculate the mean of sample payoffs to get estimate of the expected payoff
    # discount expected payoff at the risk free rate

    

    return