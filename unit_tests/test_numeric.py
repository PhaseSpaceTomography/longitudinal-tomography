import unittest
from Numeric import *


class TestPhysics(unittest.TestCase):

    def test_newton_basic(self):
        xvalue = 2.1
        ans = newton(self._function, self._dfunction, xvalue, None, None)
        self.assertAlmostEqual(ans, np.pi)

    def test_newton_iter_limit(self):
        xvalue = 2.1
        with self.assertRaises(Exception):
            ans = newton(self._function, self._dfunction,
                         xvalue, None, None, max_iter=5)

    def test_linear_fit(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 1.5, 1.7, 1.7, 1.9, 2.0, 2.33])
        a, b, _, _, _, _ = lin_fit(x, y)
        self.assertAlmostEqual(a, 1.1767857142857143)
        self.assertAlmostEqual(b, 0.18535714285714283)

    @staticmethod
    def _function(xvalue, parameters=None, rf_turn=None):
        return np.sin(xvalue)

    @staticmethod
    def _dfunction(xvalue, parameters=None, rf_turn=None):
        return np.cos(xvalue)

