# ================
# DLC Exceptions
# ================


class LibraryNotFound(Exception):
    pass


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


class RawDataImportError(Exception):
    pass


class ArgumentError(Exception):
    pass


class NegativeIndexError(IndexError):
    pass

# ===========================
# Time space Exceptions
# ===========================


class RebinningError(Exception):
    pass


class WaterfallReducedToZero(Exception):
    pass


class FilteredProfilesError(Exception):
    pass


class WaterfallError(Exception):
    pass


class ProfileChargeNotCalculated(Exception):
    pass

# ============================
# MapInfo exceptions
# ============================


class EnergyBinningError(Exception):
    pass


class EnergyLimitsError(Exception):
    pass


class PhaseLimitsError(Exception):
    pass


class MapCreationError(Exception):
    pass


# ============================
# PARTICLE TRACKING EXCEPTIONS
# ============================

class InvalidParticleError(Exception):
    pass


class TrackingError(Exception):
    pass


class SelfFieldTrackingError(Exception):
    pass

# ============================
# TOMOGRAPHY EXCEPTIONS
# ============================


class PhaseSpaceReducedToZeroes(Exception):
    pass


class XPOutOfImageWidthError(Exception):
    pass


class CoordinateImportError(Exception):
    pass


class CoordinateError(Exception):
    pass
