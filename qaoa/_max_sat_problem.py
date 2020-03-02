# Created by Anton Bozhedarov

import numpy as np


def generate_3sat(variable_number, clause_number):
    """
    Returns a random 3-SAT problem in CNF form
    (see https://en.wikipedia.org/wiki/Boolean_satisfiability_problem#3-satisfiability)

    Args:
        variable_number (int): The number of boolean variables of 3-SAT problem.
        clause_number (int): The number of clauses of CNF form of 3-SAT problem.

    Returns:
        numpy array: random 3-SAT problem in CNF form
    """
    list_of_clauses = []
    counter = 0

    while counter < clause_number:
        new_item = np.random.choice(range(1, variable_number+1), size=3, replace=False)
        new_item=np.sort(new_item)
        negation = np.random.choice([-1,1], size=3)
        new_item *= negation

        if new_item.tolist() not in list_of_clauses:
            list_of_clauses.append(new_item.tolist())
            counter = counter + 1

    return np.asarray(list_of_clauses)


def get_bit(z, i):
    """
    gets the i'th bit of the integer z (0 labels least significant bit)
    """
    return (z >> i) & 0x1


def max_sat_obj(z, clause_list):
    """
    Returns loss for a max SAT problem. Here we count the number of violated clauses
    """
    loss = 0
    for var_i, var_j, var_k in clause_list:
        sign_i, sing_j, sing_k = np.sign(var_i), np.sign(var_j), np.sign(var_k)
        loss += (1 - sign_i * get_bit(z, var_i)) \
                * (1 - sing_j * get_bit(z, var_j)) \
                * (1 - sing_k * get_bit(z, var_k))
    return loss
