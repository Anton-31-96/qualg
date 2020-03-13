# created by Anton Bozhedarov

import pandas as pd
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize, brute

from qualg import max_sat_obj


class QAOA_error(Exception):
    pass


class QAOA_circuit(object):

    def __init__(self, clause_list, depth, noise=None):
        self.num_qubits = np.abs(clause_list).max()
        self.clause_list = clause_list
        self.depth = depth
        self.noise = noise
        self.gate_U = self._create_c_op()
        self.gate_B = self._create_b_op()
        self.hamiltonian = self.hamiltonian()

    def evolve(self, gs, bs):
        dim = 2 ** self.num_qubits
        state = np.ones((dim, dim)) / dim # prepare qubits in state plus (use density matrix)
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

    def optimize(self, angles_0=None, optimizer='COBYLA', maxiter=10_000):
        """
        Arguments:
            angles_0 (list | None): initial guess of gammas and betas values
            optimizer (str): method for parameters optimization ('COBYLAS' | 'GridSearch')
            maxiter (int): maximum number of iteration in optimization procedure

        Returns:
            qaoa_ans (dict): contains digest of QAOA optimization
        """

        if angles_0 is None:
            angles = np.zeros(2 * self.depth)
        else:
            angles = angles_0

        bnds_g = [(0, 2 * np.pi)] * self.depth
        bnds_b = [(0, np.pi)] * self.depth
        bnds = bnds_g + bnds_b

        ans = minimize(self.expv, angles, method=optimizer, bounds=bnds, options={'maxiter': maxiter})

        exact_xs, exact_loss = self.exact_solution()
        angles = ans.x

        qaoa_ans = {'output_qaoa': self.max_bitstr(angles),
                    'state_energy': ans.fun,
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

    def _create_b_op(self):

        num_qubits = self.num_qubits
        dim = 2 ** num_qubits
        gate = np.zeros((dim, dim))
        for i in range(1, num_qubits+1):
            gate = gate + get_x_j(i, num_qubits)

        circuit_noise = self.noise

        def expm(beta, noise=circuit_noise):
            return Evolution(la.expm(-1j * beta * gate), noise=noise)

        return expm

    def _create_c_op(self):

        ham = self.hamiltonian()
        circuit_noise = self.noise

        def expm(gamma, noise=circuit_noise):
            return Evolution(la.expm(-1j * gamma * ham), noise=noise)

        return expm

    def hamiltonian(self):
        clause_list = self.clause_list
        num_qubits = np.abs(clause_list).max()
        dim = 2 ** num_qubits
        proj = np.zeros((dim, dim))

        for clause in clause_list:
            proj_j = self.projector(clause)
            proj = proj + proj_j

        return proj

    def projector(self, idx):

        num_qubits = self.num_qubits

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


def build_qaoa(clause_list, depth, noise=None):
    return QAOA_circuit(clause_list, depth, noise)


class EvolutionError(Exception):
    pass


class Evolution(object):

    def __init__(self, matrix, noise=None):
        self.mtx = np.matrix(matrix)

        if noise is None:
            pass
        elif isinstance(noise, DepolChannel):
            pass
        else:
            raise EvolutionError("Use class of Error to initialize Evolution operator")
        self.noise = noise

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """

        new_state = self.matrix @ state @ self.matrix.H

        if self.noise is None:
            return new_state
        else:
            return self.noise | new_state

    def __str__(self):
        return str(self.matrix)

    @property
    def matrix(self):
        return self.mtx


class DepolChannel(object):

    def __init__(self, p):
        self.p = p

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """
        dim = len(state)
        return np.eye(dim)/dim * self.p + (1 - self.p) * state


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


def generate_data(num_variables, sat_list, depth, noise=None):

    data = pd.DataFrame()

    for fixed_d_clause in tqdm(sat_list):
        for clause_list in fixed_d_clause:

            cc = dm.build_qaoa(clause_list, depth, noise=noise)
            ans = cc.optimize()

            clause_density = len(clause_list) / num_variables
            ans.update({'clause_density': clause_density})
            data = data.append(ans, ignore_index=True)

    return data
