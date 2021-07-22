"""Module containing assertion functions with standardised output for
tomography

:Author(s): **Christoffer Hjertø Grindheim**
"""
from numbers import Number
from typing import Union, Type, Tuple, Any, Collection, TYPE_CHECKING

import numpy as np

from longitudinal_tomography import exceptions as expt

if TYPE_CHECKING:
    from longitudinal_tomography.tracking.machine import Machine
    from longitudinal_tomography.utils.tomo_input import Frames


# =========================================================
#                      SCALAR ASSERTIONS
# =========================================================
def assert_greater(var: Union[int, float], var_name: str,
                   limit: Union[int, float], error_class: Type[Exception],
                   extra_text: str = ''):
    """Assert scalar greater than X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be greater than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var <= limit:
        msg = _write_std_err_msg(var_name, var, limit, '>', extra_text)
        raise error_class(msg)


def assert_less(
        var: Union[int, float],
        var_name: str, limit: Union[int, float],
        error_class: Type[Exception],
        extra_text: str = ''):
    """Assert scalar less than X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be less than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var >= limit:
        msg = _write_std_err_msg(var_name, var, limit, '<', extra_text)
        raise error_class(msg)


def assert_equal(var: Union[int, float], var_name: str,
                 limit: Union[int, float], error_class: Type[Exception],
                 extra_text: str = ''):
    """Assert scalar equal to X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var != limit:
        msg = _write_std_err_msg(var_name, var, limit, '==', extra_text)
        raise error_class(msg)


def assert_not_equal(var: Union[int, float], var_name: str,
                     limit: Union[int, float], error_class: Type[Exception],
                     extra_text: str = ''):
    """Assert scalar unequal to X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be unequal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var == limit:
        msg = _write_std_err_msg(var_name, var, limit, '!=', extra_text)
        raise error_class(msg)


def assert_less_or_equal(var: Union[int, float], var_name: str,
                         limit: Union[int, float],
                         error_class: Type[Exception], extra_text: str = ''):
    """Assert scalar less than or equal to X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be less than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var > limit:
        msg = _write_std_err_msg(var_name, var, limit, '<=', extra_text)
        raise error_class(msg)


def assert_greater_or_equal(
        var: Union[int, float],
        var_name: str, limit: Union[int, float],
        error_class: Type[Exception],
        extra_text: str = ''):
    """Assert scalar greater than or equal to X.

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    limit: int, float
        Limit of which variable should be greater than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var < limit:
        msg = _write_std_err_msg(var_name, var, limit, '>=', extra_text)
        raise error_class(msg)


def assert_inrange(var: Union[int, float], var_name: str,
                   low_lim: Union[int, float], up_lim: Union[int, float],
                   error_class: Type[Exception], extra_text: str = ''):
    """Assert scalar is in range of [x, y].

    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for users reference.
    low_lim: int, float
        Lower limit of variable.
    up_lim: int, float
        Upper limit of variable.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if var < low_lim or var > up_lim:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} should be in area '
                         f'[{low_lim}, {up_lim}].')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


# =========================================================
#                      ARRAY ASSERTIONS
# =========================================================


def assert_array_not_equal(array: np.ndarray, array_name: str,
                           limit: Union[int, float],
                           error_class: Type[Exception], extra_text: str = ''):
    """Assert array not equal to X.

    This function asserts that not all
    elements of a value has a scalar value X.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    limit: int, float
        Scalar value which the array should be unequal of.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    """
    if np.all(array == limit):
        error_message = (f'\nAll elements of the array "{array_name}" '
                         f'has the unexpected value: {array[0]}.\n'
                         f'Expected value: {array_name} != {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_array_shape_equal(arrays: Tuple[np.ndarray, np.ndarray],
                             array_names: Tuple[str, str],
                             demanded_shape: Tuple[int, int],
                             extra_text: str = ''):
    """Assert that two arrays have a given shape.

    Parameters
    ----------
    arrays: tuple
        tuple should contain (ndarrayX, ndarrayY).
        ndarrays are the arrays to be asserted.
    array_names: tuple
        tuple of strings, being names of the arrays to be tested for the
        users reference.
    demanded_shape: tuple
        Demanded shape of arrays to be asserted. Tuple: (int, int).
    extra_text: string
        Extra text to be written after standard error message.
    """
    assert_greater_or_equal(len(arrays), 'number of arrays', 2,
                            AssertionError,
                            'Unable to compare arrays, since less than two '
                            'arrays are given.')
    ok = True
    counter = 0
    array_error_str = ''
    for i in range(len(arrays)):
        if demanded_shape != arrays[i].shape:
            ok = False
            array_error_str += (f'{array_names[i]} with '
                                f'shape: {arrays[i].shape}\n')
            counter += 1
    if not ok:
        error_message = (f'\ndeviation from a desired array shape '
                         f'of {demanded_shape} found in array(s):\n'
                         f'{array_error_str}'
                         f'{extra_text}')
        raise expt.UnequalArrayShapes(error_message)


def assert_array_numel_equal(arrays: Tuple[np.ndarray, np.ndarray],
                             array_names: Tuple[str, str],
                             demanded_numel: Tuple[int, int],
                             extra_text: str = ''):
    """Assert that two arrays have the given number of items

    Parameters
    ----------
    arrays: tuple
        tuple should contain (ndarrayX, ndarrayY).
        ndarrays are the arrays to be asserted.
    array_names: tuple
        tuple of strings, being names of the arrays to be tested for the
        users reference.
    demanded_numel: tuple
        Demanded shape of arrays to be asserted. Tuple: (int, int).
    extra_text: string
        Extra text to be written after standard error message.
    """
    assert_greater_or_equal(len(arrays), 'number of arrays', 2,
                            AssertionError,
                            'Unable to compare arrays, since less than two '
                            'arrays are given.')
    if isinstance(demanded_numel, Number):
        demanded_numel = (demanded_numel,)

    ok = True
    counter = 0
    array_error_str = ''
    for i in range(len(arrays)):
        if demanded_numel != arrays[i].shape:
            ok = False
            array_error_str += (f'{array_names[i]} with '
                                f'shape: {arrays[i].shape}\n')
            counter += 1
    if not ok:
        error_message = (f'\ndeviation from a desired array size '
                         f'of {demanded_numel} found in array(s):\n'
                         f'{array_error_str}'
                         f'{extra_text}')
        raise expt.UnequalArrayShapes(error_message)


def assert_array_in_range(array: np.ndarray, low_lim: Union[int, float],
                          up_lim: Union[int, float],
                          error_class: Type[Exception],
                          msg: str = '', index_offset: int = 0):
    """Assert all array elements are within a range [x, y].

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    low_lim: int, float
        Lower limit of array elements.
    up_lim: int, float
        Upper limit of array elements.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text to be written after standard error message.
    index_offset: int
        Index offset when checking equality. Only elements after the offset
        are checked.
    """
    log_arr = np.where(np.logical_or(array < low_lim, array > up_lim),
                       False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater(array: np.ndarray, limit: Union[int, float],
                         error_class: Type[Exception],
                         msg: str = '', index_offset: int = 0):
    """Assert all array elements are greater than x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    limit: int, float
        Scalar value array elements should be greater than.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text to be written after standard error message.
    index_offset: int
        Index offset when checking equality. Only elements after the offset
        are checked.
    """
    log_arr = np.where(array <= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater_eq(array: np.ndarray, limit: Union[int, float],
                            error_class: Type[Exception],
                            msg: str = '', index_offset: int = 0):
    """Assert all array elements are greater than or equal to x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    limit: int, float
        Scalar value array elements should be greater than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text to be written after standard error message.
    index_offset: int
        Index offset when checking equality. Only elements after the offset
        are checked.
    """
    log_arr = np.where(array < limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less(array: np.ndarray, limit: Union[int, float],
                      error_class: Type[Exception],
                      msg: str = '', index_offset: int = 0):
    """Assert all array elements are less than x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    limit: int, float
        Scalar value array elements should be less than.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text to be written after standard error message.
    index_offset: int
        Index offset when checking equality. Only elements after the offset
        are checked.
    """
    log_arr = np.where(array >= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less_eq(array: np.ndarray, limit: Union[int, float],
                         error_class: Type[Exception],
                         msg: str = '', index_offset: int = 0):
    """Assert all array elements are less than or equal to x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    limit: int, float
        Scalar value array elements should be less than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text to be written after standard error message.
    index_offset: int
        Index offset when checking equality. Only elements after the offset
        are checked.
    """
    log_arr = np.where(array > limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


# Checks that all elements of a logical array is true.
# If not, an error is raised.
# Used by the array assertion functions.
def _assert_log_arr(log_array_ok: np.ndarray, error_class: Type[Exception],
                    index_offset: int, msg: str):
    if not log_array_ok.all():
        error_msg = '\nError found at index: ' \
                    '{}\n'.format(
                        np.argwhere(log_array_ok is False).flatten()
                        + index_offset
                    )
        raise error_class(error_msg + msg)


# =========================================================
#                 TRACKING/PARTICLE ASSERTIONS
# =========================================================

def assert_only_valid_particles(xp: np.ndarray, n_bins: int, msg: str = ''):
    """Assert all particles are within the image width.

    An InvalidParticleError is raised if the trajectory
    of one or more particles goes outside of the image width.

    Parameters
    ----------
    xp: ndarray
        Array of particles.
    n_bins: int
        Number of bins in a profile measurement (image width).
    msg: string
        Extra text for error message.

    Raises
    ------
    InvalidParticleError: Exception
        One or mor particles has left the image.
    """
    if np.any(np.logical_or(xp >= n_bins, xp < 0)):
        err_msg = f'Invalid (lost) particle(s) was found in xp\n'
        raise expt.InvalidParticleError(err_msg + msg)


# =========================================================
#                     MACHINE ASSERTIONS
# =========================================================


def assert_fields(obj: Any, obj_name: str, needed_fields: Collection,
                  error_class: Type[Exception], msg: str = ''):
    """Assert that object contains all necessary fields.

    Parameters
    ----------
    obj: Object
        Object to be asserted.
    obj_name: string
        Name of object, for user reference.
    needed_fields: array like
        Array containing attributes which object should contain.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text for error message.
    """
    for field in needed_fields:
        if not hasattr(obj, field) or getattr(obj, field) is None:
            err_msg = f'Missing parameter "{field}" in {obj_name}.'
            if len(msg) > 0:
                err_msg += f'\n{msg}'
            raise error_class(err_msg)


def assert_machine_input(machine: 'Machine'):
    """Assert that input parameters for a machine object is valid.

    Parameters
    ----------
    machine: Machine
        Machine object to be asserted.

    Raises
    ------
    MachineParameterError: Exception
        Invalid value found for machine parameter.
    SpaceChargeParameterError: Exception
        Invalid value found for machine parameter regarding self-field
        measurements.
    """
    # Bin assertions
    assert_var_not_none(machine.dtbin, 'dtbin', expt.MachineParameterError)
    assert_greater(machine.dtbin, 'dtbin', 0, expt.MachineParameterError,
                   'NB: dtbin is the difference of time in bin')

    assert_var_not_none(machine.dturns, 'dturns', expt.MachineParameterError)
    assert_greater(machine.dturns, 'dturns', 0, expt.MachineParameterError,
                   'NB: dturns is the number of machine turns'
                   'between each measurement')

    # Assertions: profile to be reconstructed
    if (machine.filmstart is None or machine.filmstop is None or
            machine.filmstep is None):
        raise expt.MachineParameterError(
            'film start, film stop and filmstep cannot be None, but '
            'must be positive integers, within the number of profiles.')
    else:
        assert_greater_or_equal(machine.filmstart, 'film start',
                                0, expt.MachineParameterError)
        assert_greater_or_equal(machine.filmstop, 'film stop',
                                machine.filmstart, expt.MachineParameterError)
        assert_less_or_equal(abs(machine.filmstep), 'film step',
                             abs(machine.filmstop - machine.filmstart + 1),
                             expt.MachineParameterError)
        assert_not_equal(machine.filmstep, 'film step', 0,
                         expt.MachineParameterError)

    # Reconstruction parameter assertions
    assert_var_not_none(machine.niter, 'niter', expt.MachineParameterError)
    assert_greater(machine.niter, 'niter', 0, expt.MachineParameterError,
                   'NB: niter is the number of iterations of the '
                   'reconstruction process')

    assert_var_not_none(machine.snpt, 'snpt', expt.MachineParameterError)
    assert_greater(machine.snpt, 'snpt', 0, expt.MachineParameterError,
                   'NB: snpt is the square root '
                   'of number of tracked particles in each cell'
                   'of the reconstructed phase space.')

    # Reference frame assertions
    assert_var_not_none(machine.machine_ref_frame, 'Machine ref. frame',
                        expt.MachineParameterError)
    assert_greater_or_equal(machine.machine_ref_frame, 'Machine ref. frame',
                            0, expt.MachineParameterError)
    assert_var_not_none(machine.beam_ref_frame, 'Beam ref. frame',
                        expt.MachineParameterError)
    assert_greater_or_equal(machine.beam_ref_frame, 'Beam ref. frame',
                            0, expt.MachineParameterError)

    # Machine parameter assertion
    assert_var_not_none(
        machine.h_num, 'Harmonic number', expt.MachineParameterError)
    assert_greater(
        machine.h_num, 'Harmonic number', 0, expt.MachineParameterError)

    assert_var_not_none(machine.h_ratio, 'Harmonic ratio',
                        expt.MachineParameterError)
    assert_greater_or_equal(machine.h_ratio, 'Harmonic ratio',
                            0, expt.MachineParameterError)

    assert_var_not_none(machine.b0, 'B field (B0)', expt.MachineParameterError)
    assert_greater(machine.b0, 'B field (B0)', 0, expt.MachineParameterError)

    assert_var_not_none(machine.mean_orbit_rad, 'mean orbit radius',
                        expt.MachineParameterError)
    assert_greater(machine.mean_orbit_rad, 'mean orbit radius',
                   0, expt.MachineParameterError)
    assert_var_not_none(machine.bending_rad, 'Bending radius',
                        expt.MachineParameterError)
    assert_greater(machine.bending_rad, 'Bending radius',
                   0, expt.MachineParameterError)
    assert_var_not_none(machine.e_rest, 'rest energy',
                        expt.MachineParameterError)
    assert_greater(machine.e_rest, 'rest energy',
                   0, expt.MachineParameterError)

    # Space charge parameter assertion
    if machine.pickup_sensitivity is not None:
        assert_greater_or_equal(machine.pickup_sensitivity,
                                'pick-up sensitivity',
                                0, expt.SpaceChargeParameterError)
    if machine.g_coupling is not None:
        assert_greater_or_equal(machine.g_coupling, 'g_coupling',
                                0, expt.SpaceChargeParameterError,
                                'NB: g_coupling:'
                                'geometrical coupling coefficient')


def assert_frame_inputs(frame: 'Frames'):
    """Assert that frame parameters are valid, and that raw data will be
    correctly shaped to waterfall.

    Parameters
    ----------
    frame: Frame
        Frame object to be asserted.

    Raises
    ------
    InputError: Exception
        An invalid frame parameter has been found.
    """
    assert_greater(frame.nframes, 'nr of frames', 0, expt.InputError)
    assert_inrange(frame.skip_frames, 'skip frames',
                   0, frame.nframes - 1, expt.InputError)
    assert_greater(frame.nbins_frame, 'frame length', 0, expt.InputError)
    assert_inrange(frame.skip_bins_start, 'skip bins start',
                   0, frame.nbins_frame - 1, expt.InputError,
                   'The number of skipped bins in for a frame cannot'
                   'exceed the number of bins in that same frame')
    assert_inrange(frame.skip_bins_end, 'skip bins end',
                   0, frame.nbins_frame - 1, expt.InputError,
                   'The number of skipped bins in for a frame cannot'
                   'exceed the number of bins in that same frame')
    assert_less(frame.skip_bins_start + frame.skip_bins_end,
                'total bins skipped', frame.nbins_frame, expt.InputError,
                'The number of skipped bins in for a frame cannot'
                'exceed the number of bins in that same frame')
    assert_greater_or_equal(frame.rebin, 're-binning factor',
                            1, expt.InputError)


# =========================================================
#                      INDEX ASSERTIONS
# =========================================================

def assert_index_ok(index: int, index_limit: int, wrap_around: bool = False) \
        -> int:
    """Assert index is valid.

    Assert that index is within bounds [0, index limit).
    Wrap_around for negative indices can be enabled. If this is the
    case, index -1 is points at the last index.

    Parameters
    ----------
    index: int
        Index to be tested.
    index_limit: int
        Maximum index + 1
    wrap_around: bool, optional, default=False
        Set to true to enable wrap around for negative indices.

    Returns
    -------
    index: int
        Asserted index from 0 to index_limit-1.

    Raises
    ------
    IndexError: Exception
        Index out of bounds.
    NegativeIndexError: Exception
        Negative index given without setting wrap_around=True
    """
    index = int(index)
    if index < 0:
        if not wrap_around:
            raise expt.NegativeIndexError('Index cannot be negative')
        elif index_limit + index < 0:
            raise IndexError('Index is out of bounds')
        else:
            index += index_limit
    else:
        if index >= index_limit:
            raise IndexError('Index is out of bounds')
    return index


# =========================================================
#                     ASSERTION UTILITIES
# =========================================================

# Generate standard error message for asserting scalars.
def _write_std_err_msg(var_name: str, var: Union[int, float],
                       limit: Union[int, float],
                       operator: str, extra: str) -> str:
    error_message = (f'\nInput parameter "{var_name}" has the '
                     f'unexpected value: {var}.\n'
                     f'Expected value: {var_name} {operator} {limit}.')
    error_message += f'\n{extra}'
    return error_message


def assert_var_not_none(var: Any, var_name: str, error_class: Type[Exception]):
    if var is None:
        raise error_class(f'{var_name} cannot be of type None')
