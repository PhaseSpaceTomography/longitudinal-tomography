'''Module containing assertion functions with standarised output for tomography.

:Author(s): **Christoffer Hjertø Grindheim**
'''

import numpy as np

from . import exceptions as expt

# =========================================================
#                      SCALAR ASSERTIONS
# =========================================================
def assert_greater(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar greater than X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be greater than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var <= limit:
        msg = _write_std_err_msg(var_name, var, limit, '>', extra_text)
        raise error_class(msg)


def assert_less(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar less than X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be less than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var >= limit:
        msg = _write_std_err_msg(var_name, var, limit, '<', extra_text)
        raise error_class(msg)


def assert_equal(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar equal to X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var != limit:
        msg = _write_std_err_msg(var_name, var, limit, '==', extra_text)
        raise error_class(msg)


def assert_not_equal(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar unequal to X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be unequal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var == limit:
        msg = _write_std_err_msg(var_name, var, limit, '!=', extra_text)
        raise error_class(msg)


def assert_less_or_equal(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar less than or equal to X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be less than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var > limit:
        msg = _write_std_err_msg(var_name, var, limit, '<=', extra_text)
        raise error_class(msg)


def assert_greater_or_equal(var, var_name, limit, error_class, extra_text=''):
    '''Assert scalar greater than or equal to X.
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    limit: int, float
        Limit of which variable should be greater than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    if var < limit:
        msg = _write_std_err_msg(var_name, var, limit, '>=', extra_text)
        raise error_class(msg)


def assert_inrange(var, var_name, low_lim, up_lim, error_class, extra_text=''):
    '''Assert scalar is in range of [x, y].
    
    Parameters
    ----------
    var: int, float
        Variable to be asserted.
    var_name: string
        Name of variable, for user reference.
    low_limit: int, float
        Lower limit of variable.
    up_limit: int, float
        Upper limit of variable.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
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


def assert_array_not_equal(array, array_name, limit, 
                           error_class, extra_text=''):
    '''Assert array not equal to X.
    
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
    '''
    if np.all(array == limit):
        error_message = (f'\nAll elements of the array "{array_name}" '
                         f'has the unexpected value: {array[0]}.\n'
                         f'Expected value: {array_name} != {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_array_shape_equal(arrays, array_names,
                             demanded_shape, extra_text=''):
    '''Assert that two arrays have a given shape.

    Parameters
    ----------
    arrays: tuple (ndarray, ndarray)
        Arrays to be asserted.
    array_names: tuple (string, string)
        Names of arrays, for user reference.
    demanded_shape: tuple (int, int)
        Demanded shape of arrays to be asserted.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
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


def assert_array_in_range(array, low_lim, up_lim, error_class,
                          msg='', index_offset=0):
    '''Assert all array elements are within a range [x, y].

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    low_lim: int, float
        Lower limit of array elements.
    up_lim: int, float
        Upper limit of array elements.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    log_arr = np.where(np.logical_or(array < low_lim, array > up_lim),
                       False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater(array, limit, error_class,
                         msg='', index_offset=0):
    '''Assert all array elements are greater than x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    limit: int, float
        Scalar value array elements should be greater than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    log_arr = np.where(array <= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater_eq(array, limit, error_class,
                            msg='', index_offset=0):
    '''Assert all array elements are greater than or equal to x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    limit: int, float
        Scalar value array elements should be greater than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    log_arr = np.where(array < limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less(array, limit, error_class,
                      msg='', index_offset=0):
    '''Assert all array elements are less than x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    limit: int, float
        Scalar value array elements should be less than.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    log_arr = np.where(array >= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less_eq(array, limit, error_class,
                         msg='', index_offset=0):
    '''Assert all array elements are less than or equal to x.

    Parameters
    ----------
    array: ndarray
        Array to be asserted.
    array_name: string
        Name of array, for user reference.
    limit: int, float
        Scalar value array elements should be less than or equal to.
    error_class: Exception
        Error class to be raised if assertion fails.
    extra_text: string
        Extra text to be written after standard error message.
    '''
    log_arr = np.where(array > limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


# Checks that all elements of a logical array is true.
# If not, an error is raised.
# Used by the array assertion functions.
def _assert_log_arr(log_array_ok, error_class, index_offset, msg):
    if not log_array_ok.all():
        error_msg = f'\nError found at index: ' \
            f'{np.argwhere(log_array_ok == False).flatten() + index_offset}\n'
        raise error_class(error_msg + msg)

# =========================================================
#                 TRACKING/PARTICLE ASSERTIONS
# =========================================================

def assert_only_valid_particles(xp, n_bins, msg=''):
    '''Assert all particles are within the image width.
    
    An error is raised if the trajectory of one or more particles
    goes outside of the image width.

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
    '''
    if np.any(np.logical_or(xp >= n_bins, xp < 0)):
        err_msg = f'Invalid (lost) particle(s) was found in xp\n'
        raise expt.InvalidParticleError(err_msg + msg)


# =========================================================
#                     MACHINE ASSERTIONS
# =========================================================


def assert_fields(obj, obj_name, needed_fields, error_class, msg=''):
    '''Assert that object contains all necessary fields.
    
    Parameters
    ----------
    obj: Object
        Object to be asserted.
    obj_name: string
        Name of object, for user reference.
    needed_fields: List
        List containing attributes which tha object should contain.
    error_class: Exception
        Error class to be raised if assertion fails.
    msg: string
        Extra text for error message.
    '''
    for field in needed_fields:
        if not hasattr(obj, field):
            err_msg = f'Missing parameter "{field}" in {obj_name}.'
            if len(msg) > 0:
                err_msg += f'\n{msg}' 
            raise error_class(err_msg)


def assert_machine_input(machine):
    '''Assert that input parameters for a machine object is valid.
    
    Parameters
    ----------
    machine: Machine
        Machine object to be asserted.

    Raises
    ------
    MachineParameterError: Exception
        Invalid value found for machine parameter. 
    SpaceChargeParameterError: Exeption
        Invalid value found for machine parameter regarding self-field
        measurements.
    '''
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
    assert_greater_or_equal(
        machine.h_num, 'Harmonic number', 1, expt.MachineParameterError)

    assert_var_not_none(machine.h_ratio, 'Harmonic ratio',
                        expt.MachineParameterError)
    assert_greater_or_equal(machine.h_ratio, 'Harmonic ratio',
                            1, expt.MachineParameterError)

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

def assert_frame_inputs(frame):
    '''Assert frame parameters are valid.
    
    Asserts that raw data will be correctly shaped to waterfall.

    Parameters
    ----------
    frame: Frame
        Frame to be asserted.

    Raises
    ------
    InputError: Exception
        An invalid frame parameter has been found.
    '''
    assert_greater(frame.nframes, 'nr of frames', 0, expt.InputError)
    assert_inrange(frame.skip_frames, 'skip frames',
                   0, frame.nframes-1, expt.InputError)
    assert_greater(frame.nbins_frame, 'frame length', 0, expt.InputError)
    assert_inrange(frame.skip_bins_start, 'skip bins start',
                   0, frame.nbins_frame-1, expt.InputError,
                   'The number of skipped bins in for a frame cannot'
                   'exceed the number of bins in that same frame')
    assert_inrange(frame.skip_bins_end, 'skip bins end',
                   0, frame.nbins_frame-1, expt.InputError,
                   'The number of skipped bins in for a frame cannot'
                   'exceed the number of bins in that same frame')
    assert_less(frame.skip_bins_start + frame.skip_bins_end,
                'total bins skipped', frame.nbins_frame, expt.InputError,
                'The number of skipped bins in for a frame cannot'
                'exceed the number of bins in that same frame')
    assert_greater_or_equal(frame.rebin, 're-binning factor',
                            1, expt.InputError)

# =========================================================
#                     ASSERTION UTILITIES
# =========================================================

# Generate standard error message for asserting scalars.
def _write_std_err_msg(var_name, var, limit, operator, extra):
    error_message = (f'\nInput parameter "{var_name}" has the '
                     f'unexpected value: {var}.\n'
                     f'Expected value: {var_name} {operator} {limit}.')
    error_message += f'\n{extra}'
    return error_message

def assert_var_not_none(var, var_name, error_class):
    if var is None:
        raise error_class(f'{var_name} cannot be of type None')
