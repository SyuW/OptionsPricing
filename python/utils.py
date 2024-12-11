import numpy as np
from scipy.interpolate import CubicSpline


def roundClosestPower(n, p):
    """
    Round n to the closest power of p and return it.
    """
    curr = 1
    while curr < n:
        curr *= p
    if abs(n-curr/p) < abs(n-curr):
        curr = curr // p
    return curr


def calculatePnormError(arr1, arr2, p=2):
    """
    Calculate error using mean p-norm error
    Note: both arrays should be the same shape
    """
    assert len(arr1) == len(arr2)
    return (1 / len(arr1)) * np.sum(np.abs(arr1 - arr2) ** p)


def interpolateOptionPrices(spot, stock_prices, option_prices):
    """
    Interpolate the option price as a function of stock price, and
    compute as function of spot price using the interpolation
    """
    cs = CubicSpline(stock_prices, option_prices)
    return cs(spot)


def getInput(message, assert_list=[], func=float):
    """
    Get and return user input
    """
    flag = False
    while not flag:
        try:
            user_input = input(message)
            user_input = func(user_input)
            flag = True
        except ValueError:
            print("Error: input could not be converted, please try again.")
        except Exception as e:
            print("An error occurred:", str(e), ". Please try again.")
    # if user input is a string, need to check that string is valid for later
    if func == str:
        assert user_input in assert_list
    return func(user_input)


# write all tests here
if __name__ == "__main__":
    print(roundClosestPower(63, 8))