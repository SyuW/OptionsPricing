import numpy as np
from scipy.interpolate import CubicSpline


def pnorm_error():

    return


def interpolateOptionPrices(spot, stock_prices, option_prices):

    cs = CubicSpline(stock_prices, option_prices)

    return cs(spot)


def getInput(message, assert_list=[], func=float):

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