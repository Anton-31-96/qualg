# created by Anton Bozhedarov

import numpy as np
from numpy import sqrt


class NoiseError(Exception):
    pass


class NoiseChannel(object):
    pass


class DepolChannel(NoiseChannel):

    def __init__(self, p):
        self.p = p

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """
        dim = len(state)
        return np.eye(dim)/dim * self.p + (1 - self.p) * state


# class AmplitudeDamping(NoiseChannel):
#
#     def __init__(self, p, gamma):
#         self.p = p
#         self.gamma = gamma
#
#         E_0 = np.sqrt(p) * np.matrix([[1, 0], [0, np.sqrt(1 - gamma)]])
#         E_1 = np.sqrt(p) * np.matrix([[0, np.sqrt(gamma)], [0, 0]])
#         E_2 = np.sqrt(p-1) * np.matrix([[np.sqrt(1 - gamma), 0], [0, 1]])
#         E_3 = np.sqrt(p-1) * np.matrix([[0, 0], [np.sqrt(gamma), 0]])
#         kraus_ops = [E_0, E_1, E_2, E_3]
#         self.kraus_ops = kraus_ops
#
#     def __or__(self, state):
#         for kraus_op in self.kraus_ops:
#             state += kraus_op @ state @ kraus_op.H
#         return state


class AmplitudePhaseDamping(NoiseChannel):

    def __init__(self, param_amp, param_phase, excited_state_population=0):

        if param_amp < 0:
            raise NoiseError(f"Invalid amplitude damping parameter, {param_amp} < 0")
        if param_phase < 0:
            raise NoiseError(f"Invalid phase damping parameter, {param_phase} < 0")
        if param_amp + param_phase > 1:
            raise NoiseError(f"Invalid amplitude and phase damping parameters, {param_amp} + {param_phase} > 1")
        if excited_state_population < 0:
            raise NoiseError(f"Invalid excited state population parameter, {excited_state_population} < 0")
        if excited_state_population > 1:
            raise NoiseError(f"Invalid excited state population parameter, {excited_state_population} > 1")

        # shorten variables
        p1 = excited_state_population
        a = param_amp
        b = param_phase

        # Damping to the state |0>
        A0 = sqrt(1 - p1) * np.matrix([[1, 0], [0, sqrt(1 - a - b)]])
        A1 = sqrt(1 - p1) * np.matrix([[0, sqrt(a)], [0, 0]])
        A2 = sqrt(1 - p1) * np.matrix([[0, 0], [0, sqrt(b)]])

        # Damping to the state |1>
        B0 = sqrt(p1) * np.matrix([[sqrt(1 - a - b), 0], [0, 1]])
        B1 = sqrt(p1) * np.matrix([[0, 0], [sqrt(a), 0]])
        B2 = sqrt(p1) * np.matrix([[sqrt(b), 0], [0, 0]])

        # chose non-zero operators:
        kraus_ops = [op for op in [A0, A1, A2, B0, B1, B2] if np.linalg.norm(op) > 1e-10]
        self.kraus_ops = kraus_ops

    def __or__(self, state):
        dim = int(np.log2(len(state)))
        for kraus_op in self.kraus_ops:

            # create noise operator that acts on the whole system.
            # Here we assume that error acts locally on each qubit
            kraus_op_ext = kraus_op
            for i in range(dim - 1):
                kraus_op_ext = np.kron(kraus_op_ext, kraus_op)

            state += kraus_op_ext @ state @ kraus_op_ext.H
        return state


class AmplitudeDamping(AmplitudePhaseDamping):

    def __init__(self, param_amp, excited_state_population=0):
        AmplitudePhaseDamping.__init__(self, param_amp, 0, excited_state_population)


class PhaseDamping(AmplitudePhaseDamping):

    def __init__(self, param_phase, excited_state_population=0):
        AmplitudePhaseDamping.__init__(self, param_phase, excited_state_population)
