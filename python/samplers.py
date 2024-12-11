import numpy as np
import matplotlib.pyplot as plt

from math import pow
from utils import roundClosestPower, getResultFromPayoffs


# Try to follow naming convention wherein we use greek letters when possible 

# basic brownian motion
def generateBrownianMotion(numPoints, period, numIterations, mu, sigma, method="traditional", seed=100):
    """
    Sample multiple Brownian paths. Can be done traditionally or via Brownian
    bridge. There is also an option of generating stratified paths for variance reduction.

    numPoints - number of points per path
    period - length of time for Brownian paths
    numIterations - number of replicate paths 
    mu - drift value
    sigma - volatility
    """
    assert method in ["traditional", "bridge", "stratified"]
    rng = np.random.default_rng(seed=seed)
    if method == "traditional":
        dt = period / numPoints
        Z_ij = rng.standard_normal(size=((numIterations, numPoints)))
        W_ij = np.cumsum(mu * dt + sigma * np.sqrt(dt) * Z_ij, axis=1)
    # brownian bridge will remain a bit slower until I can figure out a way
    # to vectorize the code. Implementation is taken straight out of
    # Glasserman (2003)
    elif method == "bridge":
        # round numPoints to the nearest power of two
        numPoints = roundClosestPower(numPoints, 2)
        m = int(np.log2(numPoints))
        dt = period / numPoints
        h = 2 ** m
        Z_ij = rng.standard_normal(size=((numIterations, numPoints+1)))
        W_ij = np.zeros(shape=(numIterations, numPoints+1))
        W_ij[:, -1] = mu * h * dt + sigma * np.sqrt(h * dt) * Z_ij[:, -1]
        jMax = 1
        for k in range(1, m+1):
            iMin = h // 2
            i = iMin
            l, r = 0, h
            for j in range(1, jMax+1):
                # a & b are coefficients of interpolation formula
                a = ((r - i) * W_ij[:, l] + (i - l) * W_ij[:, r]) / (r - l)
                b = sigma * np.sqrt((r - i) * (i - l) * dt / (r - l))
                W_ij[:, i] = a + b * Z_ij[:, i]
                i += h
                l += h
                r += h
            jMax *= 2
            h = iMin
    return W_ij


# geometric brownian motion
def generateGBM(numPoints, period, numIterations, S0, mu, sigma, method="traditional", seed=100):
    """
    Sample multiple geometric Brownian paths.
    """
    return S0 * np.exp(generateBrownianMotion(numPoints, period, numIterations,
                                              mu-sigma**2/2, sigma, method, seed))


# simulate a d-dimensional Brownian motion
def generateMultiDimBrownianMotion(numPoints, period, numIterations, cov, muVec, method="traditional", seed=100):
    """
    Generate a multi-dimensional Brownian motion which can have correlation between processes. 

    numPoints - number of time steps in path
    period - length of time
    numIterations - number of replicates of d Brownian paths
    cov - dxd covariance matrix
    muVec - dx1 matrix of drifts
    method - use traditional method, Brownian bridge, or terminal stratification
    """
    assert method in ["traditional", "bridge", "stratified"]
    rng = np.random.default_rng(seed=seed)
    B_ij = np.linalg.cholesky(cov)
    d = B_ij.shape[1]
    dt = period / numPoints
    multiDimPaths = []
    if method == "traditional":
        for _ in range(numIterations):
            Z_ij = rng.standard_normal(size=(d, numPoints))
            X_ij = np.cumsum(muVec[:, np.newaxis] * dt + np.sqrt(dt) * B_ij @ Z_ij,
                            axis=1)
            multiDimPaths.append(X_ij)
    elif method == "bridge":
        numPoints = roundClosestPower(numPoints, 2)
        m = int(np.log2(numPoints))
        dt = period / numPoints
        for _ in range(numIterations):
            Z_ij = rng.standard_normal(size=(d, numPoints+1))
            W_ij = np.zeros(shape=(d, numPoints+1))
            jMax = 1
            h = 2 ** m
            W_ij[:, -1] = np.sqrt(h * dt) * Z_ij[:, -1]
            for k in range(1, m+1):
                iMin = h // 2
                i = iMin
                l, r = 0, h
                for j in range(1, jMax+1):
                    a = ((r - i) * W_ij[:, l] + (i - l) * W_ij[:, r]) / (r - l)
                    b = np.sqrt((r - i) * (i - l) * dt / (r - l))
                    W_ij[:, i] = a + b * Z_ij[:, i]
                    i += h
                    l += h
                    r += h
                jMax *= 2
                h = iMin
            X_ij = np.cumsum(muVec[:, np.newaxis] * dt + np.zeros(shape=(d, numPoints+1)), axis=1) + B_ij @ W_ij
            multiDimPaths.append(X_ij)
    return multiDimPaths


# simulate a d-dimensional Geometric Brownian motion
def generateMultiDimGBM(numPoints, period, numIterations, initialPrices, cov, muVec, method, seed):
    bm = generateMultiDimBrownianMotion(numPoints, period, numIterations,
                                        cov, muVec-0.5*np.diagonal(cov)**2, method, seed)
    multiDimPaths = []
    for assetPaths in bm:
        gbmPaths = initialPrices[:, np.newaxis] * np.exp(assetPaths)
        multiDimPaths.append(gbmPaths)
    return multiDimPaths


# for testing
if __name__ == "__main__":

    # method that we want to test
    genMethod = "traditional"

    assert genMethod in ["traditional", "bridge", "stratified"]

    # test by computing the price of a European call and comparing with Black-Scholes
    # These parameters yield a theoretical call price of 6.11678
    rseed = 100
    S_0 = 50
    K = 50
    r = 0.10
    vol = 0.40
    T = 0.4167
    N = 100000
    from calculators import blackScholesPricer
    callPrice = blackScholesPricer(S=S_0, K=K, r=r, sigma=vol, T=T, q=0, type="call")
    gbm = generateGBM(10, T, N, S0=S_0, mu=r, sigma=vol, method=genMethod, seed=rseed)
    payoffs = np.exp(-r * T) * np.maximum(gbm[:,-1]-K, 0)
    print(f"The Black-Scholes price is {callPrice}")
    print(f"The price of the European call is is {getResultFromPayoffs(payoffs)}")
    
    # test by computing price of fixed-strike Asian call
    # These parameters yield a theoretical call price of 8.40878
    # rseed = 100
    # S_0 = 100
    # K = 100
    # r = 0.15
    # vol = 0.20
    # T = 1
    # N = 10000
    # gbm = generateGBM(1000, T, N, S0=S_0, mu=r, sigma=vol, method=genMethod, seed=rseed)
    # payoffs = np.exp(-r * T) * np.maximum(np.mean(gbm, axis=1)-K, 0)
    # print(f"The price of the fixed-strike Asian call is {getResultFromPayoffs(payoffs)}")

    # test the uncorrelated case of the (d=2) market model by setting rho = 0 and
    rseed = 400
    numIterations = 10000
    numPointsInPath = 10
    S_0 = np.array([100, 90])
    K = 100
    r = 0.05
    rho = 0
    sigma1, sigma2 = 0.20, 0.25
    weights = np.array([0.5, 0.5])
    muVec = np.array([r, r])
    cov = np.array([[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]])
    correlatedGBM = generateMultiDimGBM(numPointsInPath, 1, numIterations, initialPrices=S_0,
                                        cov=cov, muVec=muVec, method=genMethod, seed=rseed)
    payoffs1, payoffs2 = [], []
    for paths in correlatedGBM:
        payoffs1.append(np.exp(-r * T) * np.maximum(paths[:, -1][0] - K, 0))
        payoffs2.append(np.exp(-r * T) * np.maximum(paths[:, -1][1] - K, 0))
    
    print(f"The Black-Scholes price of stock 1 is {blackScholesPricer(S_0[0], K, r, sigma1, 1, 0, 'call')}")
    print(f"The Black-Scholes price of stock 1 is {blackScholesPricer(S_0[1], K, r, sigma2, 1, 0, 'call')}")
    print(f"The price of a call on stock 1 is {getResultFromPayoffs(payoffs1)}")
    print(f"The price of a call on stock 2 is {getResultFromPayoffs(payoffs2)}")

    # test the (d=2) market model by computing the fair price of a basket option
    rseed = 38
    numIterations = 10000
    numPointsInPath = 10
    S_0 = np.array([100, 90])
    K = 100
    r = 0.05
    rho = 0.5
    sigma1, sigma2 = 0.20, 0.25
    weights = np.array([0.5, 0.5]) # basket weights
    muVec = np.array([r, r])
    cov = np.array([[sigma1 ** 2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2 ** 2]])
    correlatedGBM = generateMultiDimGBM(numPointsInPath, 1, numIterations, initialPrices=S_0,
                                        cov=cov, muVec=muVec, method=genMethod, seed=rseed)
    payoffs = []
    for paths in correlatedGBM:
        payoffs.append(np.exp(-r * T) * np.maximum(paths[:, -1] @ weights - K, 0))
    print(f"The price of the 2-stock basket call is {getResultFromPayoffs(payoffs)}")

    vis = False
    if vis:
        # plot the paths to make sure they look right
        numPaths = 20
        numPointsInPath = 10000
        if genMethod == "bridge":
            numPointsInPath = roundClosestPower(numPointsInPath, 2) + 1
        t = np.linspace(0, 1, numPointsInPath)
        bm = generateBrownianMotion(numPointsInPath, 1, numPaths, mu=r, sigma=vol, method=genMethod, seed=rseed)
        for i in range(numPaths):
            plt.plot(t, bm[i, :])
        plt.show(); plt.clf()

        # plot the paths of multi-dimensional Brownian motion to make sure they look right. Do a simple case
        # where we have two unit-volatility stocks with a correlation rho between them
        rho = 0.95
        t = np.linspace(0, 1, numPointsInPath)
        correlatedBM = generateMultiDimBrownianMotion(numPointsInPath, 1, 1, cov=np.array([[1,rho],[rho,1]]),
                                                    muVec=np.array([1,1]), method=genMethod)[0]
        for i in range(2):
            plt.plot(t, correlatedBM[i, :], label=f"path {i+1}")
        plt.legend()
        plt.show(); plt.clf()