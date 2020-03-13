from qualg import noise

try:
    from qualg import qaoa
except ModuleNotFoundError:
    print('No module ProjectQ')

from ._routines import (generate_3sat,
                        generate_sat_list,
                        get_bit, negation,
                        max_sat_obj
                        )
