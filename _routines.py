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
