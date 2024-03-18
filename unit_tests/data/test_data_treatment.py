"""Unit-tests for the physics module.

Run as python test_data_treatment.py in console or via coverage
"""
from __future__ import annotations
import os
import unittest

import numpy as np
import numpy.testing as nptest

from .. import commons
import longitudinal_tomography.data.data_treatment as treat
import longitudinal_tomography.data.profiles as prf
import longitudinal_tomography.tracking.machine as mch

# Machine arguments based on the input file INDIVShavingC325.dat
MACHINE_ARGS = commons.get_machine_args()


class TestDataTreatment(unittest.TestCase):
    pass
