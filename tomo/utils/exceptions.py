
# ================
# Array Exceptions
# ================


class UnequalArrayShapes(Exception):
    pass

# ====================
# Numerical Exceptions
# ====================


class OverMaxIterationsError(Exception):
    pass


class ValuesOutOfBrackets(Exception):
    pass


class ArrayLengthError(Exception):
    pass


class ArrayElementsNotEqualError(Exception):
    pass

# ===========================
# Input Parameters Exceptions
# ===========================


class InputError(Exception):
    pass


class MachineParameterError(Exception):
    pass


class SpaceChargeParameterError(Exception):
    pass
