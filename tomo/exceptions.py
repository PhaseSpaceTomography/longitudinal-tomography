class TomoException(Exception):
    pass

# ================
# Array Exceptions
# ================


class UnequalArrayShapes(TomoException):
    pass

# ====================
# Numerical Exceptions
# ====================


class OverMaxIterationsError(TomoException):
    pass


class ValuesOutOfBrackets(TomoException):
    pass


class ArrayLengthError(TomoException):
    pass


class ArrayElementsNotEqualError(TomoException):
    pass

# ===========================
# Input Parameters Exceptions
# ===========================


class InputError(TomoException):
    pass


class MachineParameterError(TomoException):
    pass


class SpaceChargeParameterError(TomoException):
    pass


class RawDataImportError(TomoException):
    pass


class ArgumentError(TomoException):
    pass


class NegativeIndexError(IndexError):
    pass

# ===========================
# Time space Exceptions
# ===========================


class RebinningError(TomoException):
    pass


class WaterfallReducedToZero(TomoException):
    pass


class FilteredProfilesError(TomoException):
    pass


class WaterfallError(TomoException):
    pass


class ProfileChargeNotCalculated(TomoException):
    pass

# ============================
# MapInfo exceptions
# ============================


class EnergyBinningError(TomoException):
    pass


class EnergyLimitsError(TomoException):
    pass


class PhaseLimitsError(TomoException):
    pass


class MapCreationError(TomoException):
    pass


# ============================
# PARTICLE TRACKING EXCEPTIONS
# ============================

class InvalidParticleError(TomoException):
    pass


class TrackingError(TomoException):
    pass


class SelfFieldTrackingError(TomoException):
    pass

# ============================
# TOMOGRAPHY EXCEPTIONS
# ============================


class PhaseSpaceReducedToZeroes(TomoException):
    pass


class XPOutOfImageWidthError(TomoException):
    pass


class CoordinateImportError(TomoException):
    pass


class CoordinateError(TomoException):
    pass
