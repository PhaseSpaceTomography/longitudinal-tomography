
class TomoAssertions:

    @staticmethod
    def assert_greater(var, var_name, limit, error_class, extra_text=''):
        if var <= limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} > {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_less(var, var_name, limit, error_class, extra_text=''):
        if var >= limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} < {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_equal(var, var_name, limit, error_class, extra_text=''):
        if var != limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} == {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_not_equal(var, var_name, limit, error_class, extra_text=''):
        if var == limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} < {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_less_or_equal(var, var_name, limit,
                             error_class, extra_text=''):
        if var > limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} < {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_greater_or_equal(var, var_name, limit,
                                error_class, extra_text=''):
        if var < limit:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} < {limit}.')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)

    @staticmethod
    def assert_inrange(var, var_name, low_lim, up_lim,
                       error_class, extra_text=''):
        if var < low_lim or var > up_lim:
            error_message = (f'\nInput parameter "{var_name}" has the '
                             f'unexpected value: {var}.\n'
                             f'Expected value: {var_name} is outside area: '
                             f'[{low_lim}, {up_lim}].')
            error_message += f'\n{extra_text}'
            raise error_class(error_message)
