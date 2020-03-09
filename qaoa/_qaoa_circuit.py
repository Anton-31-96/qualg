# created by Anton Bozhedarov

import matplotlib.pyplot as plt

import projectq
from scipy.optimize import minimize, brute


# Import projectq main engine and gates
from projectq import MainEngine
from projectq.ops import Measure, H, CX, All, Allocate, MatrixGate, DaggeredGate
import numpy as np
import pdb
import scipy.sparse as sps
import scipy.sparse.linalg
from functools import reduce
from projectq.backends import CircuitDrawer, Simulator #, IBMBackend
from projectq import ops, backends
from projectq.types import Qureg

from ..noise import NoiseEngine


class QAOACircuit(object):
    def __init__(self, depth, expc, expb, qureg, z0):
        self.depth = depth
        self.expc = expc
        self.expb = expb
        self.qureg = qureg
        self.z0 = z0
        self._refresh()

    @property
    def num_param(self):
        """ number of parameters """
        return 2 * self.depth - 1

    @property
    def num_bit(self):
        """ number of vertices """
        return len(self.qureg)

    def evolve(self, bs, gs):
        """
        get U(B,bs[p-1])U(C,gs[p-2])...U(C,gs[0])U(B,bs[0])|+>
        Args:
            bs (1d array): angle parameters for U(B) matrices, 0 < bs[i] < pi.
            gs (1d array): angle parameters for U(B) matrices, 0 < gs[i] < 2*pi.
        Returns:
            1d array, final state.
        """
        qureg = self.qureg
        eng = qureg.engine
        
        All(H) | qureg # prepare |+> state

        for i in range(self.depth):
            v = self.expb(bs[i], qureg)
            if i != self.depth-1:
                v = self.expc(gs[i], qureg)

        eng.flush()
        qvec = np.array(eng.backend.cheat()[1])
        self._refresh()
        return qvec

    def _refresh(self):
        qureg = self.qureg
        ops.Measure | qureg
        for z, q in zip(self.z0, qureg):
            if int(q) != z:
                ops.X | q
        qureg.engine.flush()
        ops.Measure | qureg


def build_qaoa_circuit(clause_list, num_bit, depth, z0=None, mode='simulator', noise_model=None):
    """
    Arguments:
        clause_list (func): to construct the loss function used in C.
        num_bit (int): the number of bits.
        depth (int): the depth of circuit.
        mode (str): keyword for mode description ('simulator' | 'graphical' | 'noise')
        noise_model : model for simulation of noise
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


def c_op(clause_list):
    """
    Args:
        clause_list (list): list of clause, e.g.
            [(0.5, (1, 3, 5)), (0.2, (1, 2))] represents 0.5*Z1*Z3*Z5 + 0.2*Z1*Z2.
    Returns:
        func, func(t, qureg) for time evolution exp(-iCt).
    """
    def expb(t, qureg):
        hamiltonian = reduce(lambda x,y:x+y, [w*ops.QubitOperator(' '.join(['Z%d'%i for i in zstring])) \
                                              for w, zstring in clause_list])
        ops.TimeEvolution(t, hamiltonian=hamiltonian) | qureg
    return expb


def b_op():
    """
    Args:
        walk_mask (2darray|None, dtype=bool): the mask for allowed regions.
    Returns:
        func, func(t, qureg) for time evolution exp(-iBt).
    """
    def expb(t, qureg):
        hamiltonian = reduce(lambda x,y:x+y, [ops.QubitOperator('X%d'%i) for i in range(len(qureg))])
        ops.TimeEvolution(t, hamiltonian=hamiltonian) | qureg
    return expb


def _initialize_register(num_bit, mode='simulator', noise_model=None):
    """
    use an engine instead of current one.
    """
    import projectq.setups.default

    engine_list = []

    # create a main compiler engine with a specific backend:
    if mode == 'graphical':
        backend = CircuitDrawer()
    elif mode == 'simulator':
        backend = Simulator()
    elif mode == 'noise':
        backend = Simulator()

        # Create noise engine according to the noise_model
        if noise_model is None:
            raise
        engine_list.append(NoiseEngine(p=0.01, noise_model=noise_model))  # TODO: delete specific p later. Consider only noise_model

    else:
        raise

    eng = MainEngine(backend=backend, engine_list=engine_list)

    # initialize register
    qureg = eng.allocate_qureg(num_bit)
    return qureg
