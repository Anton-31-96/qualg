# created by Anton Bozhedarov

import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

from qualg.qaoa import max_sat_obj


class QAOA_error(Exception):
    pass


class QAOA_circuit(object):

    def __init__(self, clause_list, depth, gate_U, gate_B):
        self.num_qubits = np.abs(clause_list).max()
        self.clause_list = clause_list
        self.depth = depth
        self.gate_U = gate_U
        self.gate_B = gate_B
        self.hamiltonian = hamiltonian(clause_list)

    def evolve(self, gs, bs):
        dim = 2 ** self.num_qubits
        state = np.eye(dim) / dim # prepare qubits in state plus (use density matrix)
        for i in range(self.depth):
            state = self.gate_U(gs[i]) | state
            state = self.gate_B(bs[i]) | state
        return state

    def expv(self, x):
        """
        expectation value of qaoa hamiltonian
        """
        gs, bs = x[:self.depth], x[self.depth:]
        state = self.evolve(gs, bs)
        expval = np.trace(self.hamiltonian @ state)

        return np.real(expval)

    def max_bitstr(self, angles=None, gs=None, bs=None):
        if angles is None and gs is None and bs is None:
            raise QAOA_error("Specify angles altogether or gammas and betas separately")
        if angles is not None:
            gs, bs = angles[:self.depth], angles[self.depth:]
        state = self.evolve(gs, bs)
        decimal = np.argmax(np.abs(np.diag(state))) + 1

        return decimal

    def optimize(self, params_0=None, optimizer='COBYLA', maxiter=10_000):

        if params_0 is None:
            params = np.zeros(2 * self.depth)
        else:
            params = params_0

        ans = minimize(self.expv, params, method=optimizer, options={'maxiter': maxiter})
        exact_xs, exact_loss = self.exact_solution()
        angles = ans.x

        qaoa_ans = {'output_qaoa': self.max_bitstr(angles),
                    'state_energy' : ans.fun,
                    'exact_xs': exact_xs,
                    'exact_loss': exact_loss,
                    'angles': angles,
                    'quant_it': ans.nfev,
                    'success': ans.success
                    }
        return qaoa_ans

    def exact_solution(self):

        def loss_func(z):
            return max_sat_obj(z, self.clause_list)

        dim = 2 ** self.num_qubits
        loss_table = np.array([loss_func(z) for z in range(dim)])

        exact_x = np.argmin(loss_table)
        exact_xs = np.argwhere(loss_table == np.amin(loss_table)).flatten()
        exact_loss = loss_table[exact_x]

        return exact_xs, exact_loss


def build_qaoa(clause_list, depth):

    num_qubits = np.abs(clause_list).max()
    bexp = b_op(num_qubits)
    cexp = c_op(clause_list)

    return QAOA_circuit(clause_list, depth, cexp, bexp)


class Evolution(object):

    def __init__(self, matrix):
        self.matrix = np.matrix(matrix)

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """
        return self.matrix @ state @ self.matrix.H

    def __str__(self):
        return str(self.matrix)

    @property
    def mtx(self):
        return self.matrix


class DepolChannel(object):

    def __init__(self, p):
        self.p = p

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """
        dim = len(state)
        return np.eye(dim)/dim * self.p + (1 - self.p) * state


def b_op(num_qubits):

    dim = 2 ** num_qubits
    gate = np.zeros((dim, dim))
    for i in range(1, num_qubits+1):
        gate = gate + get_x_j(i, num_qubits)

    def expm(beta):
        return Evolution(la.expm(-1j * beta * gate))

    return expm


def get_x_j(j, num_qubits):
    Id = np.eye(2)
    X = np.array([[0, 1], [1, 0]])

    x_j = 1
    for i in range(1, j):
        x_j = np.kron(x_j, Id)
    x_j = np.kron(x_j, X)
    for i in range(j, num_qubits):
        x_j = np.kron(x_j, Id)

    return x_j


def get_z_j(j, num_qubits):
    Id = np.eye(2)
    Z = np.array([[1, 0], [0, -1]])

    z_j = 1
    for i in range(1, j):
        z_j = np.kron(z_j, Id)
    z_j = np.kron(z_j, Z)
    for i in range(j, num_qubits):
        z_j = np.kron(z_j, Id)

    return z_j


def hamiltonian(clause_list):

    num_qubits = np.abs(clause_list).max()
    dim = 2 ** num_qubits
    proj = np.zeros((dim, dim))

    for clause in clause_list:
        proj_j = projector(clause, num_qubits)
        proj = proj + proj_j

    return proj


def c_op(clause_list):

    ham = hamiltonian(clause_list)

    def expm(gamma):
        return Evolution(la.expm(-1j * gamma * ham))

    return expm


def projector(idx, num_qubits):

    p_pos = np.matrix([[1,0],[0,0]])
    p_neg = np.matrix([[0,0],[0,1]])

    Id = np.eye(2)
    proj = 1
    for i in range(1, abs(idx[0])):
        proj = np.kron(proj, Id)
    if idx[0] > 0:
        proj = np.kron(proj, p_pos)
    else:
        proj = np.kron(proj, p_neg)

    for i in range(abs(idx[0])+1, abs(idx[1])):
        proj = np.kron(proj, Id)
    if idx[1] > 0:
        proj = np.kron(proj, p_pos)
    else:
        proj = np.kron(proj, p_neg)

    for i in range(abs(idx[1])+1, abs(idx[2])):
        proj = np.kron(proj, Id)
    if idx[2] > 0:
        proj = np.kron(proj, p_pos)
    else:
        proj = np.kron(proj, p_neg)

    for i in range(abs(idx[2]), num_qubits):
        proj = np.kron(proj, Id)

    return proj
