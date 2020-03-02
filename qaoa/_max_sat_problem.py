# Created by Anton Bozhedarov

import numpy as np


def generator_3sat_new(variable_number, clause_number):
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
