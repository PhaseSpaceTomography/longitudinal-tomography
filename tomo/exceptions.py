class TomoException(Exception):
    pass

# ================
# Array Exceptions
# ================


class UnequalArrayShapes(TomoException, ValueError):
    pass

# ====================
# Numerical Exceptions
# ====================


class OverMaxIterationsError(TomoException):
    pass


class ValuesOutOfBrackets(TomoException):
    pass


class ArrayLengthError(TomoException, ValueError):
    pass


class ArrayElementsNotEqualError(TomoException, ValueError):
    pass

# ===========================
# Input Parameters Exceptions
# ===========================


class InputError(TomoException):
    pass


class MachineParameterError(TomoException, ValueError):
    pass


class SpaceChargeParameterError(TomoException, ValueError):
    pass


class RawDataImportError(TomoException):
    pass


class ArgumentError(TomoException, ValueError):
    pass


class NegativeIndexError(IndexError):
    pass

# ===========================
# Time space Exceptions
# ===========================


class RebinningError(TomoException, RuntimeError):
    pass


class WaterfallReducedToZero(TomoException, RuntimeError):
    pass


class FilteredProfilesError(TomoException, ValueError):
    pass


class WaterfallError(TomoException, ValueError):
    pass


class ProfileChargeNotCalculated(TomoException, ValueError):
    pass

# ============================
# MapInfo exceptions
# ============================


class EnergyBinningError(TomoException, ValueError):
    pass


class EnergyLimitsError(TomoException, ValueError):
    pass


class PhaseLimitsError(TomoException, ValueError):
    pass


class MapCreationError(TomoException):
    pass


# ============================
# PARTICLE TRACKING EXCEPTIONS
# ============================

class InvalidParticleError(TomoException, ValueError):
    pass


class TrackingError(TomoException, RuntimeError):
    pass


class SelfFieldTrackingError(TomoException):
    pass

# ============================
# TOMOGRAPHY EXCEPTIONS
# ============================


class PhaseSpaceReducedToZeroes(TomoException, RuntimeError):
    pass


class XPOutOfImageWidthError(TomoException, ValueError):
    pass


class CoordinateImportError(TomoException, ValueError):
    pass


class CoordinateError(TomoException, ValueError):
    pass
