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
        msg = _write_std_err_msg(var_name, var, limit, '<', extra_text)
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
        msg = _write_std_err_msg(var_name, var, limit, '>', extra_text)
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
                         msg='', index_offset=''):
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
                            msg='', index_offset=''):
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
                      msg='', index_offset=''):
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
                         msg='', index_offset=''):
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
    assert_greater(machine.dtbin, 'dtbin', 0, expt.MachineParameterError,
                   'NB: dtbin is the difference of time in bin')
    assert_greater(machine.dturns, 'dturns', 0, expt.MachineParameterError,
                   'NB: dturns is the number of machine turns'
                   'between each measurement')

    # Assertions: profile to be reconstructed
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
    assert_greater(machine.niter, 'niter', 0, expt.MachineParameterError,
                   'NB: niter is the number of iterations of the '
                   'reconstruction process')
    assert_greater(machine.snpt, 'snpt', 0, expt.MachineParameterError,
                   'NB: snpt is the square root '
                   'of #tracked particles.')

    # Reference frame assertions
    assert_greater_or_equal(machine.machine_ref_frame,
                            'machine ref. frame',
                            0, expt.MachineParameterError)
    assert_greater_or_equal(machine.beam_ref_frame, 'beam ref. frame',
                            0, expt.MachineParameterError)

    # Machine parameter assertion
    assert_greater_or_equal(machine.h_num, 'harmonic number',
                            1, expt.MachineParameterError)
    assert_greater_or_equal(machine.h_ratio, 'harmonic ratio',
                            1, expt.MachineParameterError)
    assert_greater(machine.b0, 'B field (B0)',
                   0, expt.MachineParameterError)
    assert_greater(machine.mean_orbit_rad, "mean orbit radius",
                   0, expt.MachineParameterError)
    assert_greater(machine.bending_rad, "Bending radius",
                   0, expt.MachineParameterError)
    assert_greater(machine.e_rest, 'rest energy',
                   0, expt.MachineParameterError)

    # Space charge parameter assertion
    assert_greater_or_equal(machine.pickup_sensitivity,
                            'pick-up sensitivity',
                            0, expt.SpaceChargeParameterError)
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
                   0, frame.nframes, expt.InputError)
    assert_greater(frame.nbins_frame, 'frame length', 0, expt.InputError)
    assert_inrange(frame.skip_bins_start, 'skip bins start',
                   0, frame.nbins_frame, expt.InputError)
    assert_inrange(frame.skip_bins_end, 'skip bins end',
                   0, frame.nbins_frame, expt.InputError)
    assert_greater_or_equal(frame.rebin, 're-binning factor',
                            1, expt.InputError)


def assert_parameter_arrays(machine):
    '''Assert that the calculated turn-by-turn values of a machine
    object are valid.
    
    Parameters
    ----------
    machine: Machine
        Machine object to be asserted.

    Raises
    ------
    InputError
        Raised if the number bins is invalid.
    UnequalArrayShapes
        Raised if the calculated arrays have incorrect shapes.
    '''
    assert_greater_or_equal(machine._nbins, 'profile length', 0,
                            expt.InputError,
                            f'Make sure that the sum of post- and'
                            f'pre-skip length is less'
                            f'than the frame length\n'
                            f'frame length: {machine.framelength}\n'
                            f'pre-skip length: {machine.preskip_length}\n'
                            f'post-skip length: {machine.postskip_length}')
    assert_array_shape_equal([machine.time_at_turn, machine.omega_rev0,
                              machine.phi0, machine.dphase, machine.deltaE0,
                              machine.beta0, machine.eta0, machine.e0],
                             ['time_at_turn', 'omega_re0',
                              'phi0', 'dphase', 'deltaE0',
                              'beta0', 'eta0', 'e0'],
                             (machine._calc_number_of_turns() + 1, ))


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
