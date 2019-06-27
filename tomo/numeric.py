import sys
from utils.exceptions import OverMaxIterationsError


# Newtons method for finding roots of function, using function with given parameters
# f:            function to find zero
# df:           derivative of function
# x0:           start point function
# const_params: object with static numbers needed for calculations
# epsilon:      precision of x at 0
# max_iter: maximal number of iterations allowed (exceeding
#           this number wil generate an exception)
def newton(f, df, x0, const_params, rf_turn,
           epsilon=sys.float_info.epsilon, max_iter=100):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn, const_params, rf_turn)
        if abs(fxn) < epsilon:
            return xn
        dfxn = df(xn, const_params, rf_turn)
        if dfxn == 0:
            raise ZeroDivisionError
        xn -= float(fxn) / dfxn
    raise OverMaxIterationsError("Could not find root of function, with "
                                 + str(max_iter)
                                 + "iterations, and an precision of: "
                                 + str(epsilon))
