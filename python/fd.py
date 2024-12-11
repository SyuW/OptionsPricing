import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline


# use finite differences to solve the Black-Scholes PDE to obtain option price
def finiteDifferencesPricer(K, r, sigma, q, S_max, M, T, N, type="call", style="european", version="implicit", use_log_price=False, greeks=False):
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

    if N == "auto":
        N = int((sigma ** 2) * (M ** 2) * T) + 1
        print(f"FDPricer: Number of time grid points selected automatically: N={N} to fulfill stability condition.")

    elif N < (sigma ** 2) * (M ** 2) * T and version == "explicit":
        print(f"FDPricer::Warning: solution may be unstable.",
              f"FdPricer: Number of time grid points should exceed {(sigma ** 2) * (M ** 2) * T}",
               "as necessary condition for stability of explicit method")
        
    deltaT = T / N
    deltaS = S_max / M

    # stock price increases from left-to-right: 0, deltaS, 2 * deltaS, ..., (M-1) * deltaS, S_max
    # time increases from top-to-bottom: 0, deltaT, 2 * deltaT, ..., (N-1) * deltaT, T
    grid = np.zeros((N+1, M+1)) # grid is (N + 1) x (M + 1): time on vertical, stock price on horizontal

    # helper arrays
    t_arr = np.arange(0, N+1) * deltaT
    S_arr = np.arange(0, M+1) * deltaS

    # boundary conditions on domain: [0, T] x [0, S_max]
    if style == "european":

        if type == "put":
            zero_S_bc = K * np.exp(-r * (T - t_arr))
            large_S_bc = 0 # the no-arbitrage bound is p >= max(Ke^{-rT}-S_0, 0), but we choose S_max large enough
            maturity_bc = np.maximum(K - S_arr, 0)

        elif type == "call":
            zero_S_bc = 0
            large_S_bc = S_max - K * np.exp(-r * (T - t_arr))
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

        j_arr = np.arange(1, M)

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
                    prices_at_iteration = np.maximum(prices_at_iteration, K - np.arange(1, M) * deltaS)
                elif type == "call":
                    prices_at_iteration = np.maximum(prices_at_iteration, np.arange(1, M) * deltaS - K)

            grid[i, 1:M] = prices_at_iteration

    elif version == "explicit":

        j_arr = np.arange(1, M)

        # coefficients
        a_vec = -0.5 * (r-q) * j_arr * deltaT + 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT 
        b_vec = 1 - r * deltaT - (sigma ** 2) * (j_arr ** 2) * deltaT                      
        c_vec = +0.5 * (r-q) * j_arr * deltaT + 0.5 * (sigma ** 2) * (j_arr ** 2) * deltaT

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

    return np.arange(0, M+1) * deltaS, grid[0, :]