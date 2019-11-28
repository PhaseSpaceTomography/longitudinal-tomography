import numpy as np
from .exceptions import (MachineParameterError,
                         UnequalArrayShapes,
                         InvalidParticleError,
                         MachineParameterError,
                         InputError,
                         SpaceChargeParameterError)

# =========================================================
#                      SCALAR ASSERTIONS
# =========================================================
def assert_greater(var, var_name, limit, error_class, extra_text=''):
    if var <= limit:
        msg = _write_std_err_msg(var_name, var, limit, '>', extra_text)
        raise error_class(msg)


def assert_less(var, var_name, limit, error_class, extra_text=''):
    if var >= limit:
        msg = _write_std_err_msg(var_name, var, limit, '>', extra_text)
        raise error_class(msg)


def assert_equal(var, var_name, limit, error_class, extra_text=''):
    if var != limit:
        msg = _write_std_err_msg(var_name, var, limit, '==', extra_text)
        raise error_class(msg)


def assert_not_equal(var, var_name, limit, error_class, extra_text=''):
    if var == limit:
        msg = _write_std_err_msg(var_name, var, limit, '!=', extra_text)
        raise error_class(msg)


def assert_less_or_equal(var, var_name, limit,
                         error_class, extra_text=''):
    if var > limit:
        msg = _write_std_err_msg(var_name, var, limit, '<=', extra_text)
        raise error_class(msg)


def assert_greater_or_equal(var, var_name, limit,
                            error_class, extra_text=''):
    if var < limit:
        msg = _write_std_err_msg(var_name, var, limit, '>=', extra_text)
        raise error_class(msg)


def assert_inrange(var, var_name, low_lim, up_lim,
                   error_class, extra_text=''):
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
    if np.all(array == 0):
        error_message = (f'\nAll elements of the array "{array_name}" '
                         f'has the unexpected value: {array}.\n'
                         f'Expected value: {array_name} != {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_array_shape_equal(arrays, array_names,
                            demanded_shape, extra_text=''):
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
        raise UnequalArrayShapes(error_message)


def assert_array_in_range(array, low_lim, up_lim, error_class,
                          msg='', index_offset=0):
    log_arr = np.where(np.logical_or(array < low_lim, array > up_lim),
                        False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater(array, limit, error_class,
                         msg='', index_offset=''):
    log_arr = np.where(array <= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_greater_eq(array, limit, error_class,
                         msg='', index_offset=''):
    log_arr = np.where(array < limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less(array, limit, error_class,
                         msg='', index_offset=''):
    log_arr = np.where(array >= limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)


def assert_array_less_eq(array, limit, error_class,
                         msg='', index_offset=''):
    log_arr = np.where(array > limit, False, True)
    _assert_log_arr(log_arr, error_class, index_offset, msg)



def _assert_log_arr(log_array_ok, error_class, index_offset, msg):
    if not log_array_ok.all():
        error_msg = f'\nError found at index: ' \
            f'{np.argwhere(log_array_ok == False).flatten() + index_offset}\n'
        raise error_class(error_msg + msg)

# =========================================================
#                 TRACKING/PARTICLE ASSERTIONS
# =========================================================

def assert_only_valid_particles(xp, n_bins, msg=''):
    if np.any(np.logical_or(xp >= n_bins, xp < 0)):
        err_msg = f'Invalid (lost) particle(s) was found in xp\n'
        raise InvalidParticleError(err_msg + msg)


# =========================================================
#                     MACHINE ASSERTIONS
# =========================================================


def assert_machine(machine, needed_parameters):
    for par in needed_parameters:
        if not hasattr(machine, par):
            err_msg = (f'Missing machine parameter: {par}.\n'
                       f'Have you remebered to use machine.fill()?')
            raise MachineParameterError(err_msg)


# Asserting that the input parameters from user are valid
def assert_machine_input(machine):

    # Frame assertions
    # assert_greater(machine.framecount, 'frame count', 0, InputError)
    # assert_inrange(machine.frame_skipcount, 'frame skip-count',
    #                0, machine.framecount, InputError)
    # assert_greater(machine.framelength, 'frame length', 0, InputError)
    # assert_inrange(machine.preskip_length, 'pre-skip length',
    #                0, machine.framelength, InputError)
    # assert_inrange(machine.postskip_length, 'post-skip length',
    #                0, machine.framelength, InputError)

    # Bin assertions
    assert_greater(machine.dtbin, 'dtbin', 0, InputError,
                   'NB: dtbin is the difference of time in bin')
    assert_greater(machine.dturns, 'dturns', 0, InputError,
                   'NB: dturns is the number of machine turns'
                   'between each measurement')
    # assert_inrange(machine.imin_skip, 'imin skip',
    #                0, machine.framelength, InputError)
    # assert_inrange(machine.imax_skip, 'imax skip',
    #                0, machine.framelength, InputError)
    # assert_greater_or_equal(machine.rebin, 're-binning factor',
    #                         1, InputError)

    # Assertions: profile to be reconstructed
    assert_greater_or_equal(machine.filmstart, 'film start', 0, InputError)
    assert_greater_or_equal(machine.filmstop, 'film stop',
                            machine.filmstart, InputError)
    assert_less_or_equal(abs(machine.filmstep), 'film step',
                         abs(machine.filmstop - machine.filmstart + 1),
                         InputError)
    assert_not_equal(machine.filmstep, 'film step', 0, InputError)

    # Reconstruction parameter assertions
    assert_greater(machine.niter, 'niter', 0, InputError,
                   'NB: niter is the number of iterations of the '
                   'reconstruction process')
    assert_greater(machine.snpt, 'snpt', 0, InputError,
                   'NB: snpt is the square root '
                   'of #tracked particles.')

    # Reference frame assertions
    assert_greater_or_equal(machine.machine_ref_frame,
                            'machine ref. frame',
                            0, InputError)
    assert_greater_or_equal(machine.beam_ref_frame, 'beam ref. frame',
                            0, InputError)

    # Machine parameter assertion
    assert_greater_or_equal(machine.h_num, 'harmonic number',
                            1, MachineParameterError)
    assert_greater_or_equal(machine.h_ratio, 'harmonic ratio',
                            1, MachineParameterError)
    assert_greater(machine.b0, 'B field (B0)',
                   0, MachineParameterError)
    assert_greater(machine.mean_orbit_rad, "mean orbit radius",
                   0, MachineParameterError)
    assert_greater(machine.bending_rad, "Bending radius",
                   0, MachineParameterError)
    assert_greater(machine.e_rest, 'rest energy',
                   0, MachineParameterError)

    # Space charge parameter assertion
    assert_greater_or_equal(machine.pickup_sensitivity,
                            'pick-up sensitivity',
                            0, SpaceChargeParameterError)
    assert_greater_or_equal(machine.g_coupling, 'g_coupling',
                            0, SpaceChargeParameterError,
                            'NB: g_coupling:'
                            'geometrical coupling coefficient')

# Asserting that some of the parameters calculated are valid
def assert_parameter_arrays(machine):
    assert_greater_or_equal(machine._nbins, 'profile length', 0,
                            InputError,
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


# Standard error message for asserting scalars.
def _write_std_err_msg(var_name, var, limit, operator, extra):
    error_message = (f'\nInput parameter "{var_name}" has the '
                     f'unexpected value: {var}.\n'
                     f'Expected value: {var_name} {operator} {limit}.')
    error_message += f'\n{extra}'
    return error_message