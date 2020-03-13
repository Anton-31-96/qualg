# created by Anton Bozhedarov

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
        new_item = np.random.choice(range(1, variable_number + 1), size=3, replace=False)
        new_item = np.sort(new_item)
        negation = np.random.choice([-1, 1], size=3)
        new_item *= negation

        if new_item.tolist() not in list_of_clauses:
            list_of_clauses.append(new_item.tolist())
            counter = counter + 1

    return np.asarray(list_of_clauses)


def generate_sat_list(literals_number, max_density, number_of_samples=10, seed=5) -> list:
    """
    Arguments:
        literals_number (int): how many different literals are in the sat-formula
        max_density (int): the maximum clause density in the sat_list
        number_of_samples (int): how many samples of certain clause density are in the sat_list
        seed: seed for random number generator
    """
    np.random.seed(seed)
    sat_list = []
    max_clauses = literals_number * max_density
    for clause_number in range(2, max_clauses):
        sat_density = []
        for _ in range(number_of_samples):
            sat_density.append(generate_3sat(literals_number, clause_number))
        sat_list.append(sat_density)

    return sat_list


def get_bit(z, i):
    """
    gets the i'th bit of the integer z (0 labels least significant bit)
    """
    return (z >> i) & 0x1


def negation(z, var_sign):
    """
    Returns negation of the bit z if var_sign == -1
    """

    if var_sign == 1:
        return z
    else:
        return not z


def max_sat_obj(z, clause_list):
    """
    Returns loss for a max SAT problem. Here we count the number of violated clauses
    """
    loss = 0
    for inst in clause_list:
        sign_i, sign_j, sign_k = np.sign(inst)
        var_i, var_j, var_k = np.abs(inst) - 1
        loss += negation(get_bit(z, var_i), sign_i) \
                & negation(get_bit(z, var_j), sign_j) \
                & negation(get_bit(z, var_k), sign_k)

    return loss
