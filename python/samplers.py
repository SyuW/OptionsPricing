import numpy as np
import matplotlib.pyplot as plt

from math import pow
from utils import roundClosestPower


# Try to follow naming convention wherein we use greek letters when possible 

# basic brownian motion
def generateBrownianMotion(numPoints, period, numIterations, mu, sigma, method="traditional"):
    """
    Generate several Brownian paths. Can be done traditionally, via Brownian
    bridge + option to use terminal stratification

    numPoints - number of points per path
    maturity - length of time for Brownian paths
    numIterations - number of replicate paths 
    mu - drift value
    sigma - volatility
    """
    assert method in ["traditional", "bridge", "stratified"]
    rng = np.random.default_rng(seed=42)
    if method == "traditional":
        dt = period / numPoints
        Z_ij = rng.standard_normal(size=((numIterations, numPoints)))
        W_ij = np.cumsum(mu * dt + sigma * np.sqrt(dt) * Z_ij, axis=1)
    # brownian bridge will remain a bit slower until I can figure out a way
    # to vectorize the code. Implementation is taken straight out of
    # Glasserman (2003)
    if method == "bridge":
        # round numPoints to the nearest power of two
        numPoints = roundClosestPower(numPoints, 2)
        m = int(np.log2(numPoints))
        dt = period / numPoints
        h = pow(2, m)
        Z_ij = rng.standard_normal(size=((numIterations, numPoints)))
        W_ij = np.zeros_like(Z_ij)
        jMax = 1
        for k in range(1, m+1):
            iMin = h // 2
            i = iMin
            l, r = 0, h
            for j in range(1, jMax+1):
                # coefficients of interpolation formula
                a = (r * dt - i * dt)


    return W_ij


# geometric brownian motion
def generateGBM(numPoints, period, numIterations, S0, mu, sigma):
    """
    Generate several geometric Brownian paths.
    """
    return S0 * np.exp(generateBrownianMotion(numPoints, period, numIterations,
                                              mu-sigma**2/2, sigma))


# simulate a d-dimensional Brownian motion
def generateMultiDimBrownianMotion(numPoints, period, numIterations, cov, mus):
    """
    Multidimensional market model

    cov - covariance matrix of the assets
    mus - vector of expected returns of the assets
    """
    rng = np.random.default_rng(seed=42)
    B_ij = np.linalg.cholesky(cov)
    dt = period / numPoints
    res = []
    for _ in range(numIterations):
        Z_ij = rng.standard_normal(size=(B_ij.shape[1], numPoints))
        X_ij = np.cumsum(mus[:,np.newaxis] * dt + np.sqrt(dt) * B_ij @ Z_ij,
                         axis=1)
        res.append(X_ij)
    return res


# simulate a d-dimensional Geometric Brownian motion
def generateMultiDimGBM(numPoints, period, numIterations, initialPrices, cov, mus):
    bm = generateMultiDimBrownianMotion(numPoints, period, numIterations,
                                        cov, mus-0.5*np.diagonal(cov)**2)
    res = []
    for assetPaths in bm:
        gbmPaths = initialPrices[:, np.newaxis] * np.exp(assetPaths)
        res.append(gbmPaths)
    return res


if __name__ == "__main__":
    t = np.linspace(0, 2, 10000)
    # bm = generateGeometricBrownianMotion(10000, 1, 20, S0=1, driftRate=0.2, volatility=0.4)
    

    from python.calculators import blackScholesPricer
    S_0 = 50
    K = 50
    r = 0.10
    vol = 0.40
    T = 0.4167
    callPrice = blackScholesPricer(S=S_0, K=K, r=r, sigma=vol, T=T, q=0, type="put")
    # geometric brownian motion
    N = 10000000

    bm = generateGBM(10, T, N, S0=S_0, mean=r, sigma=vol)

    payoffs = np.exp(-r * T) * np.maximum(K-bm[:,-1], 0)

    print(f"The Black-Scholes price is {callPrice}")
    print(f"The payoff is {np.average(payoffs)}")
    print(f"The fluctuation is {np.std(payoffs) / np.sqrt(N)}")
    
    # for i in range(2):
    #     plt.plot(t, bm[0][i, :])
    # plt.show()