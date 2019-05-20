import sys
import numpy as np
import scipy.special


# Newtons method for finding roots of function, using function with given parameters
def newton(f, df, x0, const_params, rf_turn,
           epsilon=sys.float_info.epsilon, max_iter=100):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn, const_params, rf_turn)
        if abs(fxn) < epsilon:
            return xn
        dfxn = df(xn, const_params, rf_turn)
        if dfxn == 0:
            raise ArithmeticError("Derived of function = 0")
        xn -= float(fxn) / dfxn
    raise ArithmeticError("Could not find root of function, with "
                          + str(max_iter)
                          + "iterations, and an precision of: "
                          + str(epsilon))


# Linear fitting
def lin_fit(x, y, sig=None):
    if sig is not None:

        if np.size(x) == np.size(y) and np.size(x) == np.size(sig):
            ndim = np.size(x)
        else:
            raise AssertionError("All elements in array are not equal"
                                 "in linear fit")

        wt = 1 / sig ** 2
        ss = np.sum(wt)
        sx = np.dot(wt, x)
        sy = np.dot(wt, y)

    else:
        if np.size(x) == np.size(y):
            ndim = np.size(x)
        else:
            raise AssertionError("All elements in array are not equal"
                                 "in linear fit")

        ss = ndim
        sx = np.sum(x)
        sy = np.sum(y)

    sx_over_ss = sx / ss
    t = x - sx_over_ss

    if sig is not None:
        t = t / sig
        b = np.dot(t / sig, y)
    else:
        b = np.dot(t, y)

    st2 = np.dot(t, t)
    b = b / st2
    a = (sy - sx * b) / ss

    siga = np.sqrt((1 + sx ** 2 / (ss * st2)) / ss)
    sigb = np.sqrt(1 / st2)

    t = y - a - b * x

    if sig is not None:
        t = t / sig
        chi2 = np.dot(t, t)
        q = scipy.special.gammaincc(0.5 * (ndim - 2), 0.5 * chi2)
    else:
        chi2 = np.dot(t, t)
        q = 1
        sigdat = np.sqrt(chi2 / (ndim - 2))
        siga = siga * sigdat
        sigb = sigb * sigdat

    return a, b, siga, sigb, chi2, q