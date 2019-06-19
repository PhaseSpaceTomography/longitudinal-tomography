import unittest
import numpy as np
from numeric import *


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

    @staticmethod
    def _function(xvalue, parameters=None, rf_turn=None):
        return np.sin(xvalue)

    @staticmethod
    def _dfunction(xvalue, parameters=None, rf_turn=None):
        return np.cos(xvalue)

