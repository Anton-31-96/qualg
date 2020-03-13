# created by Anton Bozhedarov

from ._routines import (generate_3sat,
                        generate_sat_list,
                        get_bit, negation,
                        max_sat_obj
                        )
try:
    from qualg import noise
    from qualg import qaoa
except ModuleNotFoundError:
    print('No module ProjectQ')
