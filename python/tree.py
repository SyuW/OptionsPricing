import numpy as np


def binomialTreePricer(S, K, r, sigma, T, q, n, type="call", style="european", visualize=False, greeks=False):
    """
    Cox-Ross-Rubinstein (CRR) Binomial tree pricer.
    Semantics for indexing a node in tree: price_tree[time][net up moves, stock (0) or option (1) price]

    S       : initial stock price
    K       : strike price
    r       : risk-free interest rate
    sigma   : volatility
    T       : time to expiry
    q       : continuous dividend yield
    n       : size of tree

    greeks  : compute estimates of Greek letters from tree and return
    """

    assert type in ["call", "put"]
    assert style in ["american", "european"]

    # Useful quantities to define
    t = T / n
    u = np.exp(sigma * np.sqrt(t))
    d = np.exp(-sigma * np.sqrt(t))
    a = np.exp((r-q) * t)
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
            binomial_price = np.exp(-r * t) * (p * price_tree[i+1][j+1, 1] + (1-p) * price_tree[i+1][j, 1])

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

    if greeks:
        greeks = {}

        S = price_tree[0][0, 0]
        f_00 = price_tree[0][0, 1]
        f_11 = price_tree[1][1, 1]
        f_10 = price_tree[1][0, 1]
        f_22 = price_tree[2][2, 1]
        f_21 = price_tree[2][1, 1]
        f_20 = price_tree[2][0, 1]
        h = 0.5 * S * (u ** 2 - d ** 2)

        delta = (f_11 - f_10) / (S * (u - d)) 
        gamma = ((f_22-f_21)/(S*(u**2-1)) - (f_21-f_20)/(S*(1-d**2))) / h
        theta = (f_21 - f_00) / (2 * t)

        greeks["delta"] = delta
        greeks["gamma"] = gamma
        greeks["theta"] = theta

        return price_tree[0][0, 1], greeks

    else:
        f_00 = price_tree[0][0, 1]
        return f_00


def trinomialPricer(S, K, r, sigma, T, q, n, type="call", style="european", visualize=False):
    """
    Implement trinomial tree pricer, to compare with explicit finite differences method.
    Learn about recombination in the tree?

    """
    


    return