# Created by Anton Bozhedarov

from functools import reduce

import numpy as np
from scipy.optimize import minimize, brute
from projectq import ops
from projectq.ops import QubitOperator, TimeEvolution

from ._qaoa_circuit import QAOACircuit, b_op, _initialize_register
from ._max_cut_problem import qaoa_result_digest, get_qaoa_loss


# def get_bit(z, i):
#     """
#     gets the i'th bit of the integer z (0 labels least significant bit)
#     """
#     return (z >> i) & 0x1
#
#
# def negation(z, var_sign):
#     """
#     Returns negation of the bit z if var_sign == -1
#     """
#
#     if var_sign == 1:
#         return z
#     else:
#         return not z
#
#
# def max_sat_obj(z, clause_list):
#     """
#     Returns loss for a max SAT problem. Here we count the number of violated clauses
#     """
#     loss = 0
#     for inst in clause_list:
#         sign_i, sign_j, sign_k = np.sign(inst)
#         var_i, var_j, var_k = np.abs(inst) - 1
#         loss += negation(get_bit(z, var_i), sign_i) \
#                 & negation(get_bit(z, var_j), sign_j) \
#                 & negation(get_bit(z, var_k), sign_k)
#
#     return loss


def build_qaoa_circuit_sat(clause_list, num_bit, depth, z0=None, mode='simulator', noise_model=None):
    """
    This is the copy of similar function from the _qaoa_circuit.py
    TODO: write down a new function that will track the type of problem (i.e. max-sat or max-cut).

    Arguments:
        clause_list (func): to construct the loss function used in C.
        num_bit (int): the number of bits.
        depth (int): the depth of circuit.
        mode: TODO: description
        noise_model: TODO: description
    Returns:
        QAOACircuit, the circuit run parameters.
    """

    if z0 is None:
        z0 = np.ones(num_bit, dtype='int32')
    qureg = _initialize_register(num_bit, mode=mode, noise_model=noise_model)

    # build evolution operators
    expb = b_op()
    expc = c_op(clause_list)

    return QAOACircuit(depth, expc, expb, qureg, z0)


def single_hamiltonian(clause: list):
    # separate variables number and its negation signs
    var0, var1, var2 = np.abs(clause) - 1

    # explanation for -1:
    # qubit numeration starts with 0, but clause literals
    # starts with 1 in order to not loose negation of 0 literal
    # so here 1 is subtracted to not keep 0-s qubit unused

    sign0, sign1, sign2 = [int(i) for i in np.sign(clause)]

    # create expanded Hamiltonian from the form 0.125*(1+z_i)(1+z_j)(1+z_k)
    ham0 = QubitOperator(' ')
    ham1 = QubitOperator(f'Z{var0}', sign0)
    ham2 = QubitOperator(f'Z{var1}', sign1)
    ham3 = QubitOperator(f'Z{var2}', sign2)
    ham4 = QubitOperator(f'Z{var0} Z{var1}', sign0 * sign1)
    ham5 = QubitOperator(f'Z{var0} Z{var2}', sign0 * sign2)
    ham6 = QubitOperator(f'Z{var1} Z{var2}', sign1 * sign2)
    ham7 = QubitOperator(f'Z{var0} Z{var1} Z{var2}', sign0 * sign1 * sign2)

    hams = [ham0, ham1, ham2, ham3, ham4, ham5, ham6, ham7]

    ham = QubitOperator()

    for h in hams:
        ham += 1 / 8 * h

    return ham


# This solution does not work faster unfortunately

# def single_hamiltonian(clause: list):
#     # separate variables number and its negation signs
#     var0, var1, var2 = np.abs(clause) - 1
#
#     # explanation for -1:
#     # qubit numeration starts with 0, but clause literals
#     # starts with 1 in order to not loose negation of 0 literal
#     # so here 1 is subtracted to not keep 0-s qubit unused
#
#     sign0, sign1, sign2 = [int(i) for i in np.sign(clause)]
#
#     # create expanded Hamiltonian from the form 0.125*(1+z_i)(1+z_j)(1+z_k)
#     ham_I = QubitOperator(' ')
#     ham0 = ham_I + QubitOperator(f'Z{var0}', sign0)
#     ham1 = ham_I + QubitOperator(f'Z{var1}', sign1)
#     ham2 = ham_I + QubitOperator(f'Z{var2}', sign2)
#
#     ham = ham0 * ham1 * ham2 / 8
#
#     return ham


def c_op(clause_list: np.ndarray):
    """
    Creates operator with problem Hamiltonian.

    Arguments:
        clause_list:

    Returns:
        func, func(t, qureg) for time evolution exp(-iCt).
    """

    def expb(t, qureg):
        hamiltonian = reduce(lambda x, y: x + y, [single_hamiltonian(clause) for clause in clause_list])
        ops.TimeEvolution(t, hamiltonian=hamiltonian) | qureg

    return expb


def solve_sat(clause_list, depth, x0=None, optimizer='COBYLA', max_iter=1000, mode='simulator', noise_model=None, show_answer=False, spelling=False):
    """
    Solves problem defined by SAT formula.

    Arguments:
        clause_list:
        depth:
        x0:
        optimizer:
        max_iter:
        spelling:
        show_answer:
        mode:
        noise_model:

    Returns:

    """

    num_bit = np.abs(clause_list).max()
    N = 2 ** num_bit

    def loss_func(z):
        return max_sat_obj(z, clause_list)

    valid_mask = None

    loss_table = np.array([loss_func(z) for z in range(N)])
    cc = build_qaoa_circuit_sat(clause_list, num_bit, depth=depth, mode=mode, noise_model=noise_model)

    # obtain and analyse results
    qaoa_loss, log = get_qaoa_loss(cc, loss_table, spelling=spelling)  # the expectation value of loss function

    if x0 is None:
        x0 = np.zeros(cc.num_param)

    if optimizer == 'COBYLA':
        best_x = minimize(qaoa_loss,
                          x0=x0,
                          method='COBYLA',
                          options={'maxiter': max_iter}).x
    else:
        raise
    ans = qaoa_result_digest(best_x, cc, loss_table, show_answer=show_answer)
    return ans


def show_loss_table_sat(clause_list, depth, x0=None, optimizer='COBYLA', max_iter=1000, spelling=False):
    """
    Shows loss table for sat problem.
    Temporary function for testing the algorithm
    """

    num_bit = clause_list.max()
    N = 2 ** num_bit

    def loss_func(z):
        return max_sat_obj(z, clause_list)

    loss_table = np.array([loss_func(z) for z in range(N)])

    loss_table = np.array([loss_func(z) for z in range(N)])

    return loss_table
