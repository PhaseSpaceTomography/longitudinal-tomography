from utils.exceptions import *
from tomo.machine import Machine
import numpy as np


def assert_greater(var, var_name, limit, error_class, extra_text=''):
    if var <= limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} > {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_less(var, var_name, limit, error_class, extra_text=''):
    if var >= limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} < {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_equal(var, var_name, limit, error_class, extra_text=''):
    if var != limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} == {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_not_equal(var, var_name, limit, error_class, extra_text=''):
    if var == limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} != {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_less_or_equal(var, var_name, limit,
                         error_class, extra_text=''):
    if var > limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} <= {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_greater_or_equal(var, var_name, limit,
                            error_class, extra_text=''):
    if var < limit:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} >= {limit}.')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


def assert_inrange(var, var_name, low_lim, up_lim,
                   error_class, extra_text=''):
    if var < low_lim or var > up_lim:
        error_message = (f'\nInput parameter "{var_name}" has the '
                         f'unexpected value: {var}.\n'
                         f'Expected value: {var_name} should be in area '
                         f'[{low_lim}, {up_lim}].')
        error_message += f'\n{extra_text}'
        raise error_class(error_message)


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



def assert_only_valid_particles(xp, n_bins, msg=''):
    if np.any(np.logical_or(xp >= n_bins, xp < 0)):
        err_msg = f'Invalid (lost) particle(s) was found in xp\n'
        raise InvalidParticleError(err_msg + msg)


def assert_machine(machine, needed_parameters):
    for par in needed_parameters:
        if not hasattr(machine, par):
            err_msg = (f'Missing machine parameter: {par}.\n'
                       f'Have you remebered to use machine.fill()?')
            raise MachineParameterError(err_msg)
